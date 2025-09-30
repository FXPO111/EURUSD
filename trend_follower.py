import numpy as np
import uuid
import random
from collections import deque
from order import Order, OrderSide, OrderType


class TrendSubAgent:
    def __init__(self, agent_id, capital, base_agent):
        self.agent_id = agent_id
        self.capital = capital
        self.base_agent = base_agent

        # Эмоции и стиль
        self.confidence = np.random.uniform(0.3, 1.0)
        self.fomo = np.random.uniform(0.0, 0.5)
        self.fatigue = np.random.uniform(0.0, 0.3)
        self.risk_appetite = np.random.uniform(0.5, 1.5)
        self.loss_aversion = np.random.uniform(0.5, 1.5)
        self.bias = np.random.choice(["long", "short", "neutral"], p=[0.4, 0.4, 0.2])

        # Индивидуальные параметры
        self.activity_rate = np.random.uniform(0.05, 0.25)
        self.profit_taking_threshold = np.random.uniform(1.0, 3.0)
        self.pullback_react_chance = np.random.uniform(0.1, 0.4)
        self.false_start_chance = np.random.uniform(0.05, 0.25)

        self.trend_confirm_need = np.random.randint(2, 5)
        self.trend_confirm_counter = 0

        # Позиция
        self.current_position = None  # (entry_price, side, size)
        self.hold_steps = 0

        # Новые параметры
        self.reaction_delay = random.randint(0, 5)  # рассинхронизация
        self.cooldown = 0  # фаза капитуляции

    def decide(self, market_context, order_book, current_step, crowd_bias=0.0):
        last_price = order_book.last_trade_price
        if last_price is None:
            return None

        # Капитуляция: временный уход с рынка
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        # Шанс на пропуск из-за усталости
        if random.random() < self.fatigue:
            return None

        # Задержка реакции
        if current_step % (self.reaction_delay + 1) != 0:
            return None

        # Вероятность действия (асинхронность)
        if random.random() > self.activity_rate:
            return None

        indicators = self.base_agent.calculate_indicators()
        perception = self.base_agent.perceive_market(market_context, indicators)
        volatility = indicators.get("atr", 0.01) * self.risk_appetite
        distance_from_ema = abs(last_price - indicators["ema_slow"])

        # Фальш-старт
        if self.current_position is None and random.random() < self.false_start_chance:
            return self._open_position(random.choice(["buy", "sell"]), last_price, volatility, 0.3)

        # Контртрендовый вход (ложный пробой)
        if self.current_position is None and random.random() < 0.1:
            contrarian_side = "sell" if indicators["ema_fast"] > indicators["ema_slow"] else "buy"
            return self._open_position(contrarian_side, last_price, volatility, 0.5)

        # Смена тренда
        if perception == "trend_shift":
            self.trend_confirm_counter += 1
            if self.trend_confirm_counter >= self.trend_confirm_need:
                self.bias = "long" if indicators["ema_fast"] > indicators["ema_slow"] else "short"
                self.trend_confirm_counter = 0
        else:
            self.trend_confirm_counter = 0

        # Реакция на откат
        if self.current_position and random.random() < self.pullback_react_chance:
            if self._check_pullback_exit(last_price, volatility):
                return self._exit_partial(last_price)

        # Фиксация прибыли на EMA
        if distance_from_ema > self.profit_taking_threshold * volatility and random.random() < 0.5:
            return self._exit_partial(last_price)

        # Стадность: толпа усиливает движение
        if self.current_position is None and crowd_bias > 0.3 and random.random() < self.fomo:
            return self._open_position("buy", last_price, volatility, 0.8)
        if self.current_position is None and crowd_bias < -0.3 and random.random() < self.fomo:
            return self._open_position("sell", last_price, volatility, 0.8)

        # Импульсный вход
        if self.current_position is None and perception in ["strong_trend", "weak_trend"]:
            if abs(last_price - indicators["ema_fast"]) < 0.5 * volatility:
                if random.random() < 0.3 + 0.3 * self.confidence:
                    return self._open_position(
                        "buy" if indicators["ema_fast"] > indicators["ema_slow"] else "sell",
                        last_price, volatility, 1.0
                    )

        # Частичная фиксация по времени
        if self.current_position:
            self.hold_steps += 1
            if self.hold_steps > random.randint(5, 15):
                return self._exit_partial(last_price)

        # Основная логика
        action = self.base_agent.decide_action(indicators, perception, self.bias, self.fomo, self.confidence)
        if action != "hold":
            return self._open_position(action, last_price, volatility, 1.0)

        return None

    def _open_position(self, action, price, volatility, confidence_mult):
        size = self.base_agent.calculate_position_size(volatility, price, self.confidence * confidence_mult)
        side = OrderSide.BID if action == "buy" else OrderSide.ASK
        self.current_position = (price, "long" if side == OrderSide.BID else "short", size)
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

    def _check_pullback_exit(self, last_price, volatility):
        entry_price, pos_side, _ = self.current_position
        drawdown = (last_price - entry_price) / entry_price if pos_side == "long" else (entry_price - last_price) / entry_price
        return drawdown < -volatility * random.uniform(1.0, 1.5)

    def _exit_partial(self, last_price):
        entry_price, pos_side, pos_size = self.current_position
        profit = (last_price - entry_price) if pos_side == "long" else (entry_price - last_price)

        # Адаптация эмоций
        self._update_emotions(profit)

        # Дробное закрытие
        partial_size = pos_size * random.choice([0.5, 0.33, 1.0])
        if partial_size < pos_size:
            self.current_position = (entry_price, pos_side, pos_size - partial_size)
        else:
            self.current_position = None

        close_side = OrderSide.ASK if pos_side == "long" else OrderSide.BID
        return Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=close_side,
            volume=round(partial_size, 4),
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )

    def _update_emotions(self, profit):
        if profit > 0:
            self.confidence = min(1.0, self.confidence + random.uniform(0.01, 0.05))
            if random.random() < 0.3:
                self.fomo = min(0.6, self.fomo + random.uniform(0.01, 0.03))
            self.fatigue = max(0.0, self.fatigue - random.uniform(0.01, 0.05))
        else:
            self.confidence = max(0.1, self.confidence - random.uniform(0.02, 0.06))
            self.loss_aversion = min(2.0, self.loss_aversion + random.uniform(0.01, 0.05))
            self.fatigue = min(0.5, self.fatigue + random.uniform(0.01, 0.04))

            # С вероятностью уходит в "капитуляцию"
            if random.random() < 0.1:
                self.cooldown = random.randint(3, 10)



class TrendFollowerAgent:
    def __init__(self, agent_id, capital, num_subagents=100):
        self.agent_id = agent_id
        self.capital = capital
        self.price_history = deque(maxlen=500)
        self.subagents = [
            TrendSubAgent(f"{agent_id}_sub_{i}", capital / num_subagents, self)
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

    def perceive_market(self, market_context, indicators):
        adx = indicators["adx"]
        slope = indicators["slope"]
        recent_prices = list(self.price_history)[-10:]
        recent_volatility = np.std(np.diff(recent_prices)) if len(recent_prices) > 2 else 0

        if recent_volatility > 2 * indicators["atr"] and random.random() < 0.3:
            return "trend_shift"

        if market_context and hasattr(market_context, "phase"):
            phase = market_context.phase.name
            if phase in ["trend_up", "trend_down"]:
                return "strong_trend"
            elif phase in ["volatile", "noise"]:
                return "hesitation"
            elif phase == "calm":
                return "neutral"

        if adx > 25 and abs(slope) > 0.001:
            return "strong_trend"
        elif adx > 15:
            return "weak_trend"
        else:
            return "no_trend"

    def decide_action(self, indicators, perception, bias, fomo, confidence):
        if perception == "no_trend":
            return "hold"
        if indicators["ema_fast"] > indicators["ema_slow"]:
            raw_signal = "buy"
        elif indicators["ema_fast"] < indicators["ema_slow"]:
            raw_signal = "sell"
        else:
            raw_signal = "hold"

        if bias == "long" and raw_signal == "sell" and random.random() < 0.5:
            return "hold"
        if bias == "short" and raw_signal == "buy" and random.random() < 0.5:
            return "hold"
        if fomo > 0.25 and perception == "weak_trend":
            return raw_signal
        if confidence < 0.5 and random.random() > confidence:
            return "hold"
        return raw_signal

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

        # считаем стадность
        open_long = sum(1 for s in self.subagents if s.current_position and s.current_position[1] == "long")
        open_short = sum(1 for s in self.subagents if s.current_position and s.current_position[1] == "short")
        crowd_bias = (open_long - open_short) / len(self.subagents)

        orders = []
        for sub in self.subagents:
            order = sub.decide(market_context, order_book, self.current_step, crowd_bias)
            if order:
                orders.append(order)
        return orders
