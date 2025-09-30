import uuid
import random
import numpy as np
from collections import deque
from order import Order, OrderSide, OrderType


class ReversalSubAgent:
    def __init__(self, agent_id, capital, base_agent):
        self.agent_id = agent_id
        self.capital = capital
        self.base_agent = base_agent

        self.current_position = None  # (entry_price, side, size)
        self.hold_steps = 0

        self.confidence = np.random.uniform(0.3, 1.0)
        self.patience = np.random.randint(3, 6)  # сколько подтверждений нужно
        self.reversal_counter = 0
        self.reversal_bias = None  # 'long' или 'short'

        self.ema_distance_threshold = np.random.uniform(2.0, 4.0)
        self.slope_weaken_threshold = 0.0005
        self.adx_threshold = 25
        self.activity_rate = np.random.uniform(0.2, 0.5)

    def decide(self, market_context, order_book, current_step):
        last_price = order_book.last_trade_price
        if last_price is None:
            return None

        if random.random() > self.activity_rate:
            return None

        indicators = self.base_agent.calculate_indicators()
        ema_fast = indicators["ema_fast"]
        ema_slow = indicators["ema_slow"]
        adx = indicators["adx"]
        slope = indicators["slope"]
        volatility = indicators["atr"]

        price_far = abs(last_price - ema_slow) > self.ema_distance_threshold * volatility
        trend_direction = "up" if ema_fast > ema_slow else "down"
        weakening = adx < self.adx_threshold and abs(slope) < self.slope_weaken_threshold

        if self.current_position:
            self.hold_steps += 1
            entry_price, pos_side, pos_size = self.current_position

            # Take profit: возврат к EMA
            if abs(last_price - ema_slow) < 0.5 * volatility:
                return self._exit_full(last_price)

            # Stop loss
            unrealized = (last_price - entry_price) if pos_side == "long" else (entry_price - last_price)
            if unrealized < -1.5 * volatility:
                return self._exit_full(last_price)

            # Take profit по R:R
            if unrealized > 2.0 * volatility:
                return self._exit_full(last_price)

            # Время
            if self.hold_steps > random.randint(5, 15):
                return self._exit_full(last_price)

            return None

        # Вне позиции → ищем разворот
        if price_far and weakening:
            bias = "short" if trend_direction == "up" else "long"

            if self.reversal_bias != bias:
                self.reversal_bias = bias
                self.reversal_counter = 1
            else:
                self.reversal_counter += 1

            if self.reversal_counter >= self.patience:
                self.reversal_counter = 0
                self.reversal_bias = None
                return self._open_position(bias, last_price, volatility)

        else:
            self.reversal_counter = 0
            self.reversal_bias = None

        return None

    def _open_position(self, bias, price, volatility):
        side = OrderSide.BID if bias == "long" else OrderSide.ASK
        size = self.base_agent.calculate_position_size(volatility, price, self.confidence)
        self.current_position = (price, bias, size)
        self.hold_steps = 0

        return Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=side,
            volume=round(size, 4),
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )

    def _exit_full(self, price):
        entry_price, pos_side, pos_size = self.current_position
        self.current_position = None
        close_side = OrderSide.ASK if pos_side == "long" else OrderSide.BID

        return Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=close_side,
            volume=round(pos_size, 4),
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )


class ReversalAgent:
    def __init__(self, agent_id, capital, num_subagents=100):
        self.agent_id = agent_id
        self.capital = capital
        self.price_history = deque(maxlen=500)
        self.subagents = [
            ReversalSubAgent(f"{agent_id}_sub_{i}", capital / num_subagents, self)
            for i in range(num_subagents)
        ]
        self.ema_fast_period = 12
        self.ema_slow_period = 26
        self.atr_period = 14
        self.adx_period = 14
        self.current_step = 0

    def _ema(self, data, period):
        if len(data) < period:
            return [data[-1]] * len(data)
        alpha = 2 / (period + 1)
        ema = [data[0]]
        for p in data[1:]:
            ema.append(alpha * p + (1 - alpha) * ema[-1])
        return ema

    def _atr(self, prices):
        if len(prices) < self.atr_period + 1:
            return 0.01
        high = np.array(prices) + 0.2
        low = np.array(prices) - 0.2
        close = np.array(prices)
        tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
        return float(np.mean(tr[-self.atr_period:]))

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
        price = prices[-1]
        ema_fast = self._ema(prices, self.ema_fast_period)[-1]
        ema_slow = self._ema(prices, self.ema_slow_period)[-1]
        atr = self._atr(prices)
        adx = self._adx(prices)
        slope = (ema_fast - ema_slow) / price if price > 0 else 0
        return {
            "price": price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr": atr,
            "adx": adx,
            "slope": slope,
        }

    def calculate_position_size(self, volatility, price, confidence):
        if price <= 0:
            return 1.0
        base_risk = self.capital * 0.01 * confidence
        size = base_risk / (volatility + 1e-9)
        size = min(size, (self.capital / price) * 0.1)
        return max(size, (self.capital * 0.001) / price)

    def generate_orders(self, order_book, market_context=None):
        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        self.price_history.append(last_price)
        self.current_step += 1
        orders = []
        for sub in self.subagents:
            order = sub.decide(market_context, order_book, self.current_step)
            if order:
                orders.append(order)
        return orders
