import uuid
import random
import numpy as np
from collections import deque
from order import Order, OrderSide, OrderType


class ImpulsiveSubAgent:
    def __init__(self, agent_id, capital, base_agent):
        self.agent_id = agent_id
        self.capital = capital
        self.base_agent = base_agent
        self.current_position = None  # (entry_price, bias, size, entry_step)
        self.hold_steps = 0

        # Индивидуальные параметры
        self.confidence = np.random.uniform(0.2, 1.0)
        self.patience = np.random.randint(2, 7)
        self.activity_rate = np.random.uniform(0.4, 1)
        self.behavior_type = random.choice(["impulsive", "cautious", "herd", "contrarian"])

        # Новые параметры управления позицией
        self.exit_horizon = None  # сколько максимум держим (сек)
        self.min_hold = None  # минимальное время удержания
        self.stop_frac = None  # стоп (в ATR)
        self.take_frac = None  # тейк (в ATR)
        self.cooldown = 0

    def decide(self, market_context, order_book, current_step, shared_bias=None, crowd_pressure=0.0):
        last_price = order_book.last_trade_price
        if last_price is None:
            return None

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        # Если уже есть позиция
        if self.current_position:
            return self._manage_position(last_price, market_context, current_step)

        if shared_bias is None:
            return None

        # Вероятность участия
        join_prob = self.activity_rate
        if self.behavior_type == "impulsive":
            join_prob *= 1.5
        elif self.behavior_type == "cautious":
            join_prob *= 0.5
        elif self.behavior_type == "herd":
            join_prob *= (1.0 + crowd_pressure)
        elif self.behavior_type == "contrarian":
            join_prob *= 1.5

        join_probability = join_prob * getattr(self, 'join_prob_modifier', 1.0) * (1 - np.exp(-0.05 * current_step))

        if random.random() < join_probability:
            return self._open_position(shared_bias, last_price, market_context["volatility"], current_step)
        return None

    def _open_position(self, bias, price, volatility, step):
        side = OrderSide.BID if bias == "long" else OrderSide.ASK
        size = self.base_agent.calculate_position_size(volatility, price, self.confidence)

        if size <= 0:
            return None
        side = OrderSide.BID if bias == "long" else OrderSide.ASK
        size = self.base_agent.calculate_position_size(volatility, price, self.confidence)

        # Новые параметры удержания
        self.exit_horizon = np.random.randint(120, 600)  # 2–10 минут
        self.min_hold = np.random.randint(20, 60)  # минимум 20–60 сек
        self.stop_frac = np.random.uniform(1.2, 2.0)  # стоп: 1.2–2.0 ATR
        self.take_frac = np.random.uniform(0.5, 1.0)  # тейк: 0.5–1.0 ATR

        self.current_position = (price, bias, size, step)
        self.hold_steps = 0

        return Order(str(uuid.uuid4()), self.agent_id, side, round(size, 4), None, OrderType.MARKET, None)

    def _manage_position(self, last_price, market_context, current_step):
        if self.current_position is None:
            return None
        entry_price, bias, size, entry_step = self.current_position
        self.hold_steps = current_step - entry_step
        atr = market_context["volatility"]

        # 1. Не выходим, пока не прошли min_hold секунд
        if self.hold_steps < self.min_hold:
            return None

        # 2. Стоп-лосс (цена ушла против > stop_frac * ATR)
        adverse_move = (last_price - entry_price) if bias == "long" else (entry_price - last_price)

        # Убираем зависимость от объема
        self.stop_frac = np.random.uniform(1.2, 2.0)  # стандартный стоп

        if adverse_move < -self.stop_frac * atr:
            return self._exit_full(last_price)

        # 3. Частичная фиксация (если цена ушла в нашу сторону > take_frac * ATR и прошли половину горизонта)
        if self.hold_steps > self.exit_horizon // 2 and adverse_move > self.take_frac * atr:
            if random.random() < 0.3:  # 30% шанс
                return self._exit_partial(last_price, fraction=0.5)

        # 4. Финальный выход по истечению горизонта
        if self.hold_steps >= self.exit_horizon:
            return self._exit_full(last_price)

        return None

    def _exit_full(self, price):
        entry_price, bias, size, entry_step = self.current_position
        self.current_position = None
        self.cooldown = np.random.randint(60, 300)  # отдых 1–5 мин
        close_side = OrderSide.ASK if bias == "long" else OrderSide.BID
        return Order(str(uuid.uuid4()), self.agent_id, close_side, round(size, 4), None, OrderType.MARKET, None)

    def _exit_partial(self, price, fraction=0.5):
        entry_price, bias, size, entry_step = self.current_position
        reduce_size = size * fraction
        self.current_position = (entry_price, bias, size - reduce_size, entry_step)
        close_side = OrderSide.ASK if bias == "long" else OrderSide.BID
        return Order(str(uuid.uuid4()), self.agent_id, close_side, round(reduce_size, 4), None, OrderType.MARKET, None)


class ImpulsiveAggressorAgent:
    def __init__(self, agent_id, capital, num_subagents=100):
        self.agent_id = agent_id
        self.capital = capital
        self.recent_volatility = deque(maxlen=500)
        self.price_history = deque(maxlen=500)
        self.subagents = [
            ImpulsiveSubAgent(f"{agent_id}_sub_{i}", capital / num_subagents, self)
            for i in range(num_subagents)
        ]
        self.ema_fast_period = 500
        self.ema_slow_period = 1500
        self.adx_period = 150
        self.current_step = 0
        self.bias = None
        self.last_orders_count = 0
        self.highs = deque(maxlen=1000)
        self.lows = deque(maxlen=1000)
        self.closes = deque(maxlen=1000)

    # --- risk & sizing parameters ---
    risk_base_pct = 0.01
    risk_floor_pct = 0.002
    risk_ceiling_pct = 0.02
    stop_mult = 2.0
    min_stop_frac = 0.001
    notional_cap_pct = 0.15
    lot_size = 1.0

    def calculate_position_size(self, volatility, price, confidence):
        if price <= 0 or volatility is None:
            return 0.0

        m_conf = 0.5 + confidence
        risk_pct = self.risk_base_pct * m_conf
        risk_pct = max(self.risk_floor_pct, min(risk_pct, self.risk_ceiling_pct))

        risk_budget = self.capital * risk_pct
        stop_distance = max(volatility * self.stop_mult, price * self.min_stop_frac)
        units = risk_budget / stop_distance
        max_units = (self.capital * self.notional_cap_pct) / price
        units = min(units, max_units)
        min_units = (self.capital * self.risk_floor_pct) / max(price, 1e-9)
        units = max(units, min_units)
        if self.lot_size > 0:
            units = (units // self.lot_size) * self.lot_size
        return units

    def _ema(self, data, period):
        if len(data) < period:
            return [data[-1]] * len(data)
        alpha = 2 / (period + 1)
        ema = [data[0]]
        for p in data[1:]:
            ema.append(alpha * p + (1 - alpha) * ema[-1])
        return ema

    def _atr(self, prices):
        if len(prices) < self.adx_period + 1:
            return 0.01
        high = np.array(prices) + 0.2
        low = np.array(prices) - 0.2
        close = np.array(prices)
        tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
        return float(np.mean(tr[-self.adx_period:]))

    def _adx(self, prices):
        if len(prices) < self.adx_period + 2:
            return 10.0
        diff = np.diff(prices)
        dm_plus = np.maximum(diff, 0)
        dm_minus = -np.minimum(diff, 0)
        atr = np.mean(np.abs(diff[-self.adx_period:])) + 1e-9
        di_plus = 100 * np.mean(dm_plus[-self.adx_period:]) / atr
        di_minus = 100 * np.mean(dm_minus[-self.adx_period:]) / atr
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
        return dx

    def calculate_indicators(self):
        prices = list(self.price_history)
        if not prices:
            return None
        price = prices[-1]
        ema_fast = self._ema(prices, self.ema_fast_period)[-1]
        ema_slow = self._ema(prices, self.ema_slow_period)[-1]
        adx = self._adx(prices)
        slope = (ema_fast - ema_slow) / price if price > 0 else 0
        atr = self._atr(prices)
        self.recent_volatility.append(atr)
        return {"price": price, "ema_fast": ema_fast, "ema_slow": ema_slow,
                "adx": adx, "slope": slope, "atr": atr}

    def generate_orders(self, order_book, market_context=None):
        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        self.price_history.append(last_price)
        self.current_step += 1
        indicators = self.calculate_indicators()
        if not indicators:
            return []

        trend_strength = abs(indicators["slope"])
        adx_value = indicators["adx"]
        atr_value = indicators["atr"]

        momentum = 0.0
        if len(self.price_history) > 1:
            momentum = (last_price - self.price_history[-2]) / (atr_value + 1e-9)

        # --- контртренд: сопротивление ---
        self.bias = None
        if abs(momentum) > 1.0:
            self.bias = "short" if momentum > 0 else "long"
        elif adx_value < 30 and trend_strength < 0.002:
            self.bias = "long" if indicators["ema_fast"] < indicators["ema_slow"] else "short"

        orders = []
        market_ctx = {"volatility": indicators["atr"]}
        crowd_pressure = self.last_orders_count / len(self.subagents)

        for sub in self.subagents:
            if self.bias:
                pressure_factor = min(1.0, abs(momentum) / 2.0)  # быстрее наращиваем сопротивление
                sub.join_prob_modifier = pressure_factor
            else:
                sub.join_prob_modifier = 0.1

            order = sub.decide(market_ctx, order_book, self.current_step,
                               shared_bias=self.bias, crowd_pressure=crowd_pressure)
            if order:
                orders.append(order)

        # лимит агрессии
        if self.bias and abs(momentum) > 1.0:
            max_per_step = max(1, int(len(self.subagents) * 0.4))  # до 40% субагентов
        else:
            max_per_step = max(1, int(len(self.subagents) * 0.1))

        if len(orders) > max_per_step:
            orders = random.sample(orders, max_per_step)

        self.last_orders_count = len(orders)
        return orders
