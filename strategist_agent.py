# strategist_agent.py
# Институциональный исполнитель с независимыми саб-агентами.
# Поведение: равномерное "доливание" до дедлайна микрослайсами без бурстов.

import uuid, random, time, math
from collections import deque, defaultdict
from typing import List, Optional, Dict, Any
import numpy as np
from order import Order, OrderSide, OrderType


try:
    from order import Order, OrderSide, OrderType, TICK
except Exception:
    class OrderSide: BID="BUY"; ASK="SELL"
    class OrderType: MARKET="MARKET"; LIMIT="LIMIT"
    class Order:
        def __init__(self, order_id, agent_id, side, volume, price, order_type, ttl):
            self.order_id, self.agent_id = order_id, agent_id
            self.side, self.volume, self.price = side, volume, price
            self.order_type, self.ttl = order_type, ttl
    TICK = 0.01

def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _safe(v, d):
    return v if v is not None else d


class ExecSubAgent:
    """
    Независимый саб-исполнитель.
    Поведение: равномерное «доливание» до целевой длительности 30–60 мин.
    Остаток делится на микрослайсы. MARKET редкие и дозируются токен-бакетом.
    """

    def __init__(self, agent_id: str, capital_share: float, side_bias: float = 0.0):
        self.agent_id = agent_id
        self.capital  = float(capital_share)
        self.side_bias = float(_clip(side_bias, -1.0, 1.0))

        # базовая роль и реакции среды
        self.archetype = random.choices(
            ["provider", "momentum", "contrarian", "balanced"],
            weights=[0.2, 0.3, 0.2, 0.3]
        )[0]
        self.vol_sensitivity    = random.uniform(0.5, 1.5)
        self.spread_sensitivity = random.uniform(0.5, 1.5)
        self.skip_prob          = random.uniform(0.05, 0.20)

        # направление мандата
        base_side = OrderSide.BID if random.random() < (0.5 + 0.25*self.side_bias) else OrderSide.ASK
        self.mandate_side = base_side

        # тайминги: запуск с лагом, «официальный» end_ts вторичен
        now = time.time()
        self.start_ts = now + random.uniform(0.0, 600.0)           # до 10 мин задержка
        self.end_ts   = self.start_ts + random.uniform(3*3600, 8*3600)

        # целевая длительность исполнения
        self.exec_min_s  = 30*60
        self.exec_max_s  = 60*60
        self.exec_goal_s = random.uniform(self.exec_min_s, self.exec_max_s)

        # схема прогресса (для расчёта дефицита)
        self.scheme = random.choices(["VWAP", "TWAP", "POV"], weights=[0.4, 0.3, 0.3])[0]

        # нотационал и лимиты
        self._ref_px     = None
        self.total_qty   = None
        self.cap_to_qty_ratio = random.uniform(0.25, 0.75)
        self.inv_limit   = None

        # риск
        self.regime = "normal"
        self._last_regime_check = 0.0
        self.inventory   = 0.0
        self.filled_abs  = 0.0
        self.slippage_ewma = 0.0
        self.impact_ewma   = 0.0

        # «медленный» режим
        self.pov_target    = random.uniform(0.015, 0.040)  # низкое участие
        self.slice_cap_pct = 0.004                          # 0.4% оборота 30с на слайс
        self.lam_lo, self.lam_hi = 0.08, 0.35               # редкие события
        self.dt_lo,  self.dt_hi  = 0.60, 2.20
        self.max_pending = 96

        # сайд-лок, чтобы стая не пинговала сторону
        self.side_lock_until = 0.0
        self._last_side = self.mandate_side

        # очередь будущих слайсов
        self.pending_flow: deque = deque()  # (eta, side, vol, is_mkt, price, ttl)

        # будильник планировщика
        self.next_wakeup = self.start_ts + random.uniform(2.0, 5.0)

        # --- троттлинг только для MARKET (токен-бакет) ---
        self.mkt_token_rate = 0.12      # ~1 MARKET раз в ~8.3 c
        self.mkt_token_cap  = 2.0       # запас токенов
        self._mkt_tokens = 0.0
        self._mkt_last_refill = time.time()

    # ---------- helpers ----------
    def _init_notional(self, best_bid, best_ask):
        if self._ref_px is None:
            mid = None
            if best_bid is not None and best_ask is not None:
                mid = 0.5 * (best_bid + best_ask)
            self._ref_px = _safe(mid, 100.0)
            self.total_qty = max(50.0, float(self.capital * self.cap_to_qty_ratio / max(1e-6, self._ref_px)))
            self.inv_limit = max(100.0, 1.5 * self.total_qty)

    def _update_regime(self, feat: Dict[str, float]):
        now = time.time()
        if now - self._last_regime_check < 1.0:
            return
        self._last_regime_check = now
        vol = _safe(feat.get("vol"), 0.0)
        spread = _safe(feat.get("spread"), TICK)
        if vol > 1.5 and spread > 2*TICK:
            self.regime = "high_vol"
        elif vol < 0.6 and spread <= 2*TICK:
            self.regime = "low_vol"
        else:
            self.regime = "normal"

    def _scheme_target_frac(self, now: float) -> float:
        if now <= self.start_ts:
            return 0.0
        t = (now - self.start_ts) / max(1e-9, self.exec_goal_s)
        if t >= 1.0:
            return 1.0
        if self.scheme == "TWAP":
            return _clip(t, 0.0, 1.0)
        if self.scheme == "VWAP":
            return _clip(0.5 * (1 + math.tanh(3*(t-0.5))), 0.0, 1.0)
        return _clip(0.8*t, 0.0, 1.0)

    def _dir(self, best_bid: Optional[float], best_ask: Optional[float]) -> str:
        side = self.mandate_side
        inv_ratio = abs(self.inventory) / max(1e-9, self.inv_limit)
        if inv_ratio > 1.1:
            side = OrderSide.ASK if self.mandate_side == OrderSide.BID else OrderSide.BID
        if self.regime == "high_vol" and random.random() < 0.10:
            side = OrderSide.ASK if side == OrderSide.BID else OrderSide.BID
        if self.regime == "low_vol" and random.random() < 0.05:
            side = OrderSide.ASK if side == OrderSide.BID else OrderSide.BID

        now = time.time()
        if now < self.side_lock_until:
            return self._last_side
        if side != self._last_side:
            self.side_lock_until = now + random.uniform(20.0, 60.0)
        self._last_side = side
        return side

    def _participation_cap(self, feat: Dict[str, float]) -> float:
        spread = _safe(feat.get("spread"), 0.0)
        vol    = _safe(feat.get("vol"), 0.0)
        liq    = _safe(feat.get("liq_per_sec"), 0.0)
        cap = 0.10
        s_ticks = int(round(spread / max(TICK, 1e-9))) if spread else 0
        cap *= (1.0 - 0.05*max(0, s_ticks-2))
        cap *= (1.0 / (1.0 + 1.5*vol))
        cap *= (1.0 + 0.3*math.tanh(liq))
        cap *= (1.0 - _clip(0.5*abs(self.slippage_ewma) + 0.2*math.tanh(self.impact_ewma/1000.0), 0.0, 0.6))
        return _clip(cap, 0.02, 0.25)

    def _should_act_now(self, now: float) -> bool:
        if now < self.start_ts:
            return False
        if random.random() < self.skip_prob * (0.6 if self.regime == "low_vol" else 1.0):
            return False
        return now >= self.next_wakeup

    def _schedule_next(self, now: float):
        self.next_wakeup = now + random.uniform(2.0, 5.0)

    def _refill_mkt_tokens(self, now: float):
        dt = max(0.0, now - self._mkt_last_refill)
        self._mkt_last_refill = now
        self._mkt_tokens = min(self.mkt_token_cap, self._mkt_tokens + self.mkt_token_rate * dt)

    # ---------- feedback ----------
    def on_fill(self, price, qty, side, slippage=0.0):
        sign = +1.0 if side == OrderSide.BID else -1.0
        self.inventory += sign * qty
        if side == self.mandate_side:
            self.filled_abs += qty
        else:
            self.filled_abs = max(0.0, self.filled_abs - qty)
        self.slippage_ewma = 0.9*self.slippage_ewma + 0.1*float(slippage)
        self.impact_ewma   = 0.9*self.impact_ewma   + 0.1*abs(qty)

    # ---------- planner ----------
    def _plan_drip(
        self, now: float, feat: Dict[str, Any],
        best_bid: Optional[float], best_ask: Optional[float],
        remaining: float, liq_turnover: float,
        deficit: float, t_left: float, side: str
    ):
        pov_cap = self._participation_cap(feat)
        pov     = _clip(self.pov_target, 0.01, pov_cap)

        # длительность по цели, а не по end_ts
        t_elapsed   = max(0.0, now - self.start_ts)
        t_rem_goal  = max(1.0, self.exec_goal_s - t_elapsed)

        # скорости: по цели и по участию
        rate_deadline = remaining / t_rem_goal                    # ед/с, чтобы растянуть до goal
        rate_pov      = pov * max(0.0, liq_turnover) / 30.0       # ед/с по участию
        rate = rate_deadline if rate_pov <= 0 else min(rate_deadline, rate_pov)
        # мягкий пол, чтобы не «умирать» при низкой ликвидности
        rate = max(rate, remaining / max(60.0, t_left + 600.0))

        # планируем объём на ближайшие минуты
        horizon = min(t_left + 10*60.0, self.exec_goal_s)
        need = min(remaining, max(0.0, rate * horizon))
        if need <= 0:
            return

        # лимит размера одного слайса
        slice_cap = max(1.0, self.slice_cap_pct * liq_turnover)

        # низкая «спешка»
        hurry_t   = _clip(300.0 / max(1.0, t_left), 0.0, 1.0)
        hurry_def = _clip(deficit / max(1.0, self.total_qty), 0.0, 1.0)
        hurry     = 0.3*hurry_t + 0.7*hurry_def

        # редкая подача событий
        lam = _clip(0.4 + 1.2*math.tanh(liq_turnover/2000.0) + 0.6*hurry, self.lam_lo, self.lam_hi)

        eta = now + random.uniform(0.10, 0.40)
        mid = 0.5*(best_bid+best_ask) if (best_bid is not None and best_ask is not None) else None

        while need > 0 and len(self.pending_flow) < self.max_pending:
            dt  = random.expovariate(lam)
            eta += min(self.dt_hi, max(self.dt_lo, dt))

            base = min(slice_cap, need)
            mu = math.log(max(1.0, base*0.7))
            vol  = max(1.0, min(base, random.lognormvariate(mu, 0.30)))

            # MARKET редки даже под конец
            p_mkt = _clip(0.04 + 0.18*hurry, 0.02, 0.25)
            is_mkt = (random.random() < p_mkt)

            price, ttl = None, 0.0
            if not is_mkt:
                if side == OrderSide.BID:
                    ref = best_bid if best_bid is not None else ((mid - TICK) if mid is not None else 100.0 - TICK)
                    px = ref - (TICK if random.random() < 0.3 else 0.0)
                    if best_ask is not None:
                        px = min(px, round(best_ask - TICK, 5))  # не залезаем в ask
                    price = round(px, 5)
                else:
                    ref = best_ask if best_ask is not None else ((mid + TICK) if mid is not None else 100.0 + TICK)
                    px = ref + (TICK if random.random() < 0.3 else 0.0)
                    if best_bid is not None:
                        px = max(px, round(best_bid + TICK, 5))  # не залезаем в bid
                    price = round(px, 5)
                ttl = random.uniform(2.0, 10.0)

            self.pending_flow.append((eta, side, float(vol), is_mkt, price, float(ttl)))
            need -= vol

    # ---------- main ----------
    def generate_orders(
        self, feat: Dict[str, float],
        best_bid: Optional[float], best_ask: Optional[float],
        liq_window_turnover: float
    ) -> List[Order]:
        orders: List[Order] = []
        now = time.time()

        self._init_notional(best_bid, best_ask)
        self._refill_mkt_tokens(now)

        # всё выполнено — чистим просроченное и выходим
        if self.filled_abs >= _safe(self.total_qty, 0.0):
            while self.pending_flow and self.pending_flow[0][0] <= now:
                self.pending_flow.popleft()
            return orders

        self._update_regime(feat)

        # исполняем готовые микрослайсы
        while self.pending_flow and self.pending_flow[0][0] <= now:
            eta, s, v, is_mkt, price, ttl = self.pending_flow.popleft()
            if is_mkt:
                # троттлинг MARKET: если нет токена — отложить
                self._refill_mkt_tokens(now)
                if self._mkt_tokens < 1.0:
                    eta = now + random.uniform(0.8, 2.5)
                    self.pending_flow.append((eta, s, v, True, None, ttl))
                    continue
                self._mkt_tokens -= 1.0
                orders.append(Order(
                    order_id=uuid.uuid4().hex, agent_id=self.agent_id,
                    side=s, volume=v, price=None, order_type=OrderType.MARKET, ttl=None
                ))
            else:
                # анти-пересечение через спред
                if s == OrderSide.BID and (best_ask is not None) and price >= best_ask:
                    price = round(best_ask - TICK, 5)
                if s == OrderSide.ASK and (best_bid is not None) and price <= best_bid:
                    price = round(best_bid + TICK, 5)
                orders.append(Order(
                    order_id=uuid.uuid4().hex, agent_id=self.agent_id,
                    side=s, volume=v, price=price, order_type=OrderType.LIMIT,
                    ttl=max(1, int(round(ttl)))
                ))

        # планируем по будильнику
        if self._should_act_now(now):
            target_frac = self._scheme_target_frac(now)
            target_done = target_frac * self.total_qty
            deficit = max(0.0, target_done - self.filled_abs)

            remaining = max(0.0, self.total_qty - self.filled_abs)
            t_left = max(1.0, self.end_ts - now)
            side = self._dir(best_bid, best_ask)

            self._plan_drip(
                now, feat, best_bid, best_ask,
                remaining=remaining,
                liq_turnover=liq_window_turnover,
                deficit=deficit, t_left=t_left, side=side
            )

            self._schedule_next(now)

        return orders




# ===== Координатор множества сабов =====
class InstitutionalExecutor:
    """
    Координатор независимых сабов.
    - Распределяет капитал между сабами.
    - Считает простые метрики среды.
    - Агрегирует ордера сабов.
    - Проксирует on_order_filled.
    """

    def __init__(self, agent_id: str, capital: float, num_subagents: int = 30):
        self.agent_id = agent_id
        self.capital = float(capital)
        self.num_subagents = int(max(5, num_subagents))

        # Дирихле-распределение долей
        raw = np.random.dirichlet([random.uniform(0.6, 1.6) for _ in range(self.num_subagents)])
        shares = [float(self.capital * w) for w in raw]

        side_bias = 0.0
        self.subagents: List[ExecSubAgent] = []
        for i, cap_i in enumerate(shares):
            sa = ExecSubAgent(f"{agent_id}_sub_{i}", cap_i, side_bias=side_bias)
            self.subagents.append(sa)

        # Метрики среды
        self.price_hist = deque(maxlen=1800)
        self.trade_buf  = deque(maxlen=12000)
        self.ewma_liq   = 0.0
        self.ewma_vol   = 0.0
        self.last_window_turnover = 0.0

        # буфер соответствия order_id → саб
        self._last_orders_map: Dict[str, ExecSubAgent] = {}

    # ---- приём fills от движка ----
    def on_order_filled(self, order_id: str, price: float, qty: float, side: OrderSide, slippage: float = 0.0):
        sa = self._last_orders_map.pop(order_id, None)
        if sa is not None:
            sa.on_fill(price, qty, side, slippage)

    # ---- метрики среды ----
    def _ingest_trades(self, order_book):
        # Если у движка есть recent trades — можно подключить.
        # Здесь оставляем лёгкий декор, чтобы не ломать совместимость.
        return

    def _features(self, order_book):
        # Получаем лучшую цену из книги
        try:
            best_bid = order_book._best_bid_price()
        except Exception:
            best_bid = None
        try:
            best_ask = order_book._best_ask_price()
        except Exception:
            best_ask = None

        spread = 0.0
        if best_bid is not None and best_ask is not None:
            spread = max(TICK, round(best_ask - best_bid, 5))

        # Заглушки метрик, если нет реальных данных
        liq_per_sec = max(0.1, self.ewma_liq)
        liq_ewma    = max(0.1, self.ewma_liq if self.ewma_liq else 1.0)
        vol_base    = max(0.1, self.ewma_vol if self.ewma_vol else 1.0)
        vol_now     = 1.0

        feat = {
            "spread": spread,
            "liq_per_sec": liq_per_sec,
            "liq_ewma": liq_ewma,
            "vol": vol_now,
            "vol_base": vol_base,
        }
        return feat, best_bid, best_ask

    # ---- генерация агрегированных ордеров ----
    def generate_orders(self, order_book, market_context=None, **kwargs) -> List[Order]:
        self._ingest_trades(order_book)
        feat, best_bid, best_ask = self._features(order_book)

        orders: List[Order] = []

        # отдаём контекст каждому сабу
        for sa in self.subagents:
            sub_orders = sa.generate_orders(
                feat=feat,
                best_bid=best_bid,
                best_ask=best_ask,
                liq_window_turnover=self.last_window_turnover
            ) or []
            orders.extend(sub_orders)
            for o in sub_orders:
                self._last_orders_map[o.order_id] = sa

        # мягкий колпак на долю маркетов в сумме (маскировка чрезмерной агрессии)
        if orders:
            liq_z = feat["liq_per_sec"] / max(1e-9, feat["liq_ewma"])
            vol_z = feat["vol"] / max(1e-9, feat["vol_base"])
            cap_mkt = _clip(0.55 + 0.15*math.tanh(liq_z) - 0.20*math.tanh(vol_z), 0.25, 0.75)
            m_orders = [o for o in orders if o.order_type == OrderType.MARKET]
            if len(m_orders) > int(cap_mkt * len(orders)):
                overflow = len(m_orders) - int(cap_mkt * len(orders))
                for o in m_orders[-overflow:]:
                    if o.side == OrderSide.BID and best_bid is not None:
                        o.order_type, o.price, o.ttl = OrderType.LIMIT, round(best_bid, 5), random.randint(1, 3)
                    elif o.side == OrderSide.ASK and best_ask is not None:
                        o.order_type, o.price, o.ttl = OrderType.LIMIT, round(best_ask, 5), random.randint(1, 3)

        return orders
