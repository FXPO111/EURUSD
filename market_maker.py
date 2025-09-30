import time 
import uuid
import random
import sqlite3
from candle_db import init_db

from collections import deque
from datetime import datetime

import numpy as np
from order import Order, OrderSide, OrderType
from order import quant

from mm_logger import log_summary_minute  # создадим эту функцию

LOG_FILE = "mm_pnl_log.txt"

# ---------------------------------------------------------
#  Константы
# ---------------------------------------------------------
TOP_LEVELS        = 1           # котируем лучшую цену
DEEP_LEVELS       = 2           # 2 более глубоких уровней
ICEBERG_LEVELS    = 1           # TTL=1, маленький объём
VAR_LIMIT         = -20_500_000    # дневной лимит $ per MM
MACRO_EVENTS      = {(13,30), (18,0)}  # примеры макро-паузы
MACRO_PAUSE       = 120         # секунда паузы вокруг макро-событий
ROLLING_PNL_SIZE  = 200
MAX_INV_RATIO     = 0.02        # 5% капитала максимально в позиции

# Тайминги и сессии рынка (UTC)
SESSIONS = [
    ("asia",      0, 6),
    ("pause_af",  6, 8),
    ("frankfurt", 8, 10),
    ("pause_fl", 10, 11),
    ("london",   11, 16),
    ("pause_ln", 16, 17),
    ("newyork",  17, 22),
    ("closed",   22, 24),
]

def log_summary_minute(agent, mid_price):
    mtm = agent.inventory * mid_price + agent.cash - agent.capital
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    max_pnl = max(agent.pnl_hist) if agent.pnl_hist else 0
    min_pnl = min(agent.pnl_hist) if agent.pnl_hist else 0

    log_line = (
        f"[{now}] "
        f"mode={agent.mode.upper():8} "
        f"cash={agent.cash:,.2f} "
        f"inv={agent.inventory:,.2f} "
        f"mid={mid_price:.5f} "
        f"PnL={mtm:,.2f} "
        f"(max={max_pnl:,.2f}, min={min_pnl:,.2f})"
    )

    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

def get_market_session():
    now = time.gmtime(time.time())
    h = now.tm_hour
    for name, start, end in SESSIONS:
        if start <= h < end:
            return name
    return "closed"

def _market(aid, side, vol):
    """Быстрый маркет-ордер."""
    return Order(
        uuid.uuid4().hex,
        aid,
        side,
        vol,
        price=None,
        order_type=OrderType.MARKET,
        ttl=None,
    )

# ---------------------------------------------------------
class VolEstimator:
    def __init__(self):
        self.window = deque(maxlen=40)
        self.sigma  = 0.0
    def update(self, mid):
        self.window.append(mid)
        if len(self.window) > 2:
            self.sigma = float(np.std(np.diff(self.window)))

# ---------------------------------------------------------
class AdvancedMarketMaker:
    def __init__(self, agent_id: str, capital: float):
        self.agent_id    = agent_id
        self.capital     = capital
        self.cash        = capital
        self.inventory   = 0.0
        self.vol_est     = VolEstimator()
        self.pnl_hist    = deque(maxlen=ROLLING_PNL_SIZE)
        self.active_ord  = {}
        self.ttl         = 5
        self.last_session = None
        self.session_pause_start = None
        self.local_high = None
        self.local_low = None
        self.local_mid = None
        self.recent_range = None
        self.mode = 'normal'  # normal / cautious / halt
        self.last_log_ts = 0
        self.cooldown_until = 0

    def update_market_structure(self, conn):
        now = int(time.time())
        from_ts = now - 120 * 60
        to_ts = now

        from candle_db import load_candles
        candles = load_candles(conn, '1m', from_ts, to_ts)
        if not candles:
            self.local_high = None
            self.local_low = None
            self.local_mid = None
            self.recent_range = None
            return

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        self.local_high = max(highs)
        self.local_low = min(lows)
        self.local_mid = (self.local_high + self.local_low) / 2
        self.recent_range = self.local_high - self.local_low

    def _macro_pause(self):
        now = time.time()
        utc = time.gmtime(now)
        for h,m in MACRO_EVENTS:
            target = time.mktime(time.struct_time((utc.tm_year,utc.tm_mon,utc.tm_mday,h,m,0,0,0,0)))
            if abs(target-now) <= MACRO_PAUSE:
                return True
        return False

    def _check_session_pause(self):
        current_session = get_market_session()
        if self.last_session != current_session:
            self.session_pause_start = time.time()
            self.last_session = current_session
        if self.session_pause_start is None:
            return False
        if time.time() - self.session_pause_start < MACRO_PAUSE:
            return True
        else:
            self.session_pause_start = None
            return False

    def _compute_spread(self, mid_price):
        base = 0.0007  # Базовый спред
        sigma = self.vol_est.sigma  # Волатильность
        inv_bias = abs(self.inventory) / (self.capital / 2) * 0.0015  # Сдвиг в зависимости от инвентаря

        # Рассчитываем спред с учетом волатильности и инвентаря
        spread = base + sigma * 6 + inv_bias

        # Устанавливаем минимальный спред в 1 цент
        spread = max(spread, 0.01)

        # Корректировка спреда в зависимости от рыночной сессии
        session = self.last_session or get_market_session()
        if session == "asia":
            spread *= 1.8
        elif session == "frankfurt":
            spread *= 1.2
        elif session == "london":
            spread *= 1.0
        elif session == "newyork":
            spread *= 1.1
        else:
            spread *= 1.1

        # Проверка на слишком большие позиции
        inv_ratio = self.inventory * mid_price / self.capital if self.capital else 0
        if abs(inv_ratio) > MAX_INV_RATIO:
            spread *= 1.8  # Увеличиваем спред, если позиция слишком велика

        # Если агент в режиме "осторожности", увеличиваем спред
        if self.mode == 'cautious':
            spread *= 2.5
        # В режиме "остановки" увеличиваем спред еще сильнее
        elif self.mode == 'halt':
            spread *= 10

        # Убедимся, что спред не будет больше разумного значения
        return max(0.01, min(spread, 0.01))

    def cancel_expired(self):
        now = time.time()
        drop = [oid for oid,info in self.active_ord.items() if now - info['ts'] > self.ttl]
        for oid in drop:
            self.active_ord.pop(oid, None)

    def generate_orders(self, order_book, market_context=None):
        if time.time() < self.cooldown_until:
            return []

        if self.mode == 'halt' and time.time() >= self.cooldown_until:
            self.mode = 'normal'

        if self._macro_pause() or self._check_session_pause() or self.mode == 'halt':
            self.active_ord.clear()
            return []

        self.cancel_expired()

        snap = order_book.get_order_book_snapshot(depth=1)
        if not snap['bids'] or not snap['asks']:
            return []

        best_bid = snap['bids'][0]['price']
        best_ask = snap['asks'][0]['price']
        mid = (best_bid + best_ask) / 2.0
        self.vol_est.update(mid)

        # Проверка изменения рыночной цены и удаление старых лимитных ордеров
        if self.local_mid and abs(mid - self.local_mid) > 0.01:  # если цена изменилась больше чем на 1 цент
            self.cancel_all_orders()  # Удаление всех старых лимитных ордеров

        self.local_mid = mid  # Обновление средней цены

        # Вычисление нового спреда
        spread = self._compute_spread(mid)

        base_vol_dollars = self.capital * (0.01 + random.uniform(-0.004, 0.004))
        top_vol = base_vol_dollars / mid
        deep_vol = top_vol * 1.5
        ice_vol = top_vol * 0.3

        orders = []
        price_levels = []

        aggression_threshold = 0.15
        target_bias = None
        if self.local_high and self.local_low and self.recent_range > 0:
            price_pos = (mid - self.local_low) / self.recent_range
            if price_pos > 1 - aggression_threshold:
                target_bias = OrderSide.ASK
            elif price_pos < aggression_threshold:
                target_bias = OrderSide.BID

        if target_bias and self.mode == 'normal':
            ping_vol = max(top_vol * 0.3, 5)
            for i in range(3):
                price_shift = (i + 1) * spread * 0.2
                price = mid + price_shift if target_bias == OrderSide.ASK else mid - price_shift
                order = Order(
                    uuid.uuid4().hex,
                    self.agent_id,
                    target_bias,
                    ping_vol,
                    quant(price),
                    OrderType.LIMIT,
                    ttl=1,
                    metadata={"aggressive": True}
                )
                orders.append(order)

        price_levels.append((OrderSide.BID, mid - spread / 2))
        price_levels.append((OrderSide.ASK, mid + spread / 2))
        for i in range(1, DEEP_LEVELS + 1):
            price_levels.append((OrderSide.BID, mid - spread / 2 - i * 0.015))
            price_levels.append((OrderSide.ASK, mid + spread / 2 + i * 0.015))
        for i in range(ICEBERG_LEVELS):
            price_levels.append((OrderSide.BID, mid - spread / 2 - 0.002 * i))
            price_levels.append((OrderSide.ASK, mid + spread / 2 + 0.002 * i))

        # Генерация новых ордеров
        now = time.time()
        for side, px in price_levels:
            if self.mode == 'cautious' and side == OrderSide.BID and self.inventory > 0:
                continue
            if self.mode == 'cautious' and side == OrderSide.ASK and self.inventory < 0:
                continue

            if side == OrderSide.BID and px > mid or side == OrderSide.ASK and px < mid:
                vol = ice_vol
            elif abs(px - mid) < spread:
                vol = top_vol
            else:
                vol = deep_vol

            vol = max(vol, 10)
            oid = uuid.uuid4().hex
            ttl = 1 if vol == ice_vol else self.ttl

            self.active_ord[oid] = {'side': side, 'price': px, 'vol': vol, 'ts': now}
            order = Order(oid, self.agent_id, side, vol, quant(px), OrderType.LIMIT, ttl,
                          metadata={"highlight": self.agent_id})
            orders.append(order)

        # Логирование состояния раз в минуту
        now = time.time()
        if now - self.last_log_ts >= 60:
            log_summary_minute(self, mid)
            self.last_log_ts = now

        return orders

    def cancel_order(self, oid):
        """Отмена одного ордера по ID."""
        if oid in self.active_ord:
            # Если ордер найден, удаляем его из активных ордеров
            del self.active_ord[oid]
            # Здесь можно добавить дополнительную логику для взаимодействия с OrderBook или другими компонентами
            print(f"Order {oid} cancelled.")
        else:
            print(f"Order {oid} not found.")

    def cancel_all_orders(self):
        """Отмена всех активных лимитных ордеров."""
        for oid in list(self.active_ord.keys()):
            self.cancel_order(oid)  # Теперь вызывает метод cancel_order для каждого ордера
        self.active_ord.clear()  # Очистка всех ордеров

    def on_order_filled(self, oid, price, qty, side):
        if oid not in self.active_ord:
            return []
        info = self.active_ord.pop(oid)

        if side == OrderSide.BID:
            self.cash -= price * qty
            self.inventory += qty
        else:
            self.cash += price * qty
            self.inventory -= qty

        mtm = self.inventory * price + self.cash - self.capital
        self.pnl_hist.append(mtm)

        # --- Плечо контроль ---
        notional_position = abs(self.inventory * price)
        leverage_ratio = notional_position / self.capital
        cash_ratio = abs(self.cash) / self.capital
        if leverage_ratio > 1.2 or cash_ratio > 1.2:
            self.mode = 'halt'
            self.cooldown_until = time.time() + 60
            self.cash = self.capital
            self.inventory = 0.0
            self.pnl_hist.clear()
            return []

        if mtm < VAR_LIMIT:
            self.mode = 'halt'
            self.cooldown_until = time.time() + 60
            self.cash = self.capital
            self.inventory = 0.0
            self.pnl_hist.clear()

        elif abs(mtm) > 0.5 * abs(VAR_LIMIT):
            self.mode = 'cautious'
        else:
            self.mode = 'normal'

        if self.mode == 'halt':
            hedge_side = OrderSide.ASK if self.inventory > 0 else OrderSide.BID
            hedge_qty = abs(self.inventory)
            if hedge_qty > 0:
                return [_market(self.agent_id, hedge_side, hedge_qty)]

        return []



