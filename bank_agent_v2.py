# ── bank_agent_v2.py ─────────────────────────────────────────────

import uuid
import random
import numpy as np
import logging
import os

from collections import deque
from datetime import datetime

from order_book import OrderBook
from order import Order, OrderSide, OrderType

from order import quant, TICK

os.makedirs("logs", exist_ok=True)

# Настройка логгера
bank_logger = logging.getLogger("bank_stats")
bank_logger.setLevel(logging.INFO)

if not bank_logger.handlers:
    fh = logging.FileHandler("logs/bank_stats.log")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    bank_logger.addHandler(fh)

logger = logging.getLogger("initiator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("logs/initiator.log")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# === Постоянный клиентский поток ===
class ClientFlowSimulator:
    herd_sentiment = 0.0
    news_rumor_pressure = 0.0

    @classmethod
    def update_herd_sentiment(cls, agents_mid_deltas):
        if not agents_mid_deltas:
            return
        avg = np.mean(agents_mid_deltas)
        volatility = np.std(agents_mid_deltas)
        cls.herd_sentiment = 0.9 * cls.herd_sentiment + 0.1 * np.tanh(avg / (volatility + 1e-6))

    @classmethod
    def trigger_rumor(cls):
        cls.news_rumor_pressure = random.uniform(-0.003, 0.003)

    def __init__(self, uid, behavior=None):
        self.uid = uid
        self.position = 0.0
        self.capital = 250_000_000
        self.pnl = 0.0
        self.max_drawdown = -0.1 * self.capital
        self.daily_target = 0.02 * self.capital
        self.recovery_rate = 0.005

        self.behavior = behavior or random.choices(
            ["momentum", "mean_reversion", "random"],
            weights=[0.4, 0.4, 0.2]
        )[0]

        self.prev_mid_prices = []
        self.ema_short = None
        self.ema_long = None
        self.alpha_short = 0.3
        self.alpha_long = 0.05
        self.signal_buffer = deque(maxlen=5)

        self.inertia_timer = random.randint(200, 400)

        self.levels = []
        self.detected_levels = set()
        self.level_detect_window = 20
        self.level_update_interval = 50
        self.level_check_counter = 0
        self.level_sensitivity = 0.0015

        self.startup_ticks = 5
        self.tick_counter = 0

        self.recent_pnls = deque(maxlen=20)
        self.risk_aversion = random.uniform(0.3, 0.9)
        self.confidence_decay = 1.0

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.uid:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        elif trade['sell_agent'] == self.uid:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']

        current_value = self.position * trade['price'] + self.capital
        self.pnl = current_value - 25_000_000
        self.recent_pnls.append(self.pnl)

        if self.pnl < -0.1 * self.capital:
            self.behavior = "panic"
            self.confidence_decay = 0.4
        elif self.pnl < -0.08 * self.capital:
            self.behavior = "mean_reversion"
            self.confidence_decay = 0.6
        elif self.pnl > 0.04 * self.capital:
            self.behavior = "greedy"
            self.confidence_decay = 1.4
        elif self.pnl > 0.02 * self.capital:
            self.behavior = "momentum"
            self.confidence_decay = 1.3

    def get_session_lambda(self):
        hour = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
        if 8 <= hour < 11:
            return 0.4
        elif 11 <= hour < 12:
            return 0.25
        elif 14.5 <= hour < 22:
            return 0.6
        elif 22 <= hour or hour < 6:
            return 0.6
        else:
            return 0.20

    def update_emas(self, mid_price):
        if self.ema_short is None:
            self.ema_short = mid_price
            self.ema_long = mid_price
        else:
            self.ema_short = self.alpha_short * mid_price + (1 - self.alpha_short) * self.ema_short
            self.ema_long = self.alpha_long * mid_price + (1 - self.alpha_long) * self.ema_long

    def price_momentum(self):
        if len(self.prev_mid_prices) < 5:
            return 0.0
        return self.prev_mid_prices[-1] - self.prev_mid_prices[-5]

    def proximity_to_level(self, price):
        if not self.levels:
            return 0.0
        distances = [abs(price - level) / price for level in self.levels]
        min_distance = min(distances)
        if min_distance < self.level_sensitivity:
            return 1.0 - (min_distance / self.level_sensitivity)
        return 0.0

    def detect_levels(self):
        if len(self.prev_mid_prices) < self.level_detect_window:
            return

        window = self.prev_mid_prices[-self.level_detect_window:]
        mid = len(window) // 2

        is_local_min = all(window[mid] < p for i, p in enumerate(window) if i != mid)
        is_local_max = all(window[mid] > p for i, p in enumerate(window) if i != mid)

        if is_local_min or is_local_max:
            price_level = round(window[mid], 2)
            if price_level not in self.detected_levels:
                self.levels.append(price_level)
                self.detected_levels.add(price_level)

        if len(self.levels) > 30:
            self.levels = self.levels[-30:]

    def step(self, order_book, market_context=None):
        self.tick_counter += 1
        if self.tick_counter < self.startup_ticks:
            return []

        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []

        mid_price = (best_bid + best_ask) / 2
        self.update_emas(mid_price)
        self.prev_mid_prices.append(mid_price)
        if len(self.prev_mid_prices) > 200:
            self.prev_mid_prices.pop(0)

        signal = self.ema_short - self.ema_long if self.ema_short and self.ema_long else 0
        self.signal_buffer.append(signal)
        signal_strength = np.mean(self.signal_buffer)

        self.level_check_counter += 1
        if self.level_check_counter >= self.level_update_interval:
            self.detect_levels()
            self.level_check_counter = 0

        self.inertia_timer -= 1
        if self.inertia_timer <= 0:
            self.behavior = random.choices(
                ["momentum", "mean_reversion", "random"],
                weights=[0.4, 0.4, 0.2]
            )[0]
            self.inertia_timer = random.randint(200, 400)
            self.confidence_decay = 1.0

        confidence = min(abs(signal_strength) / (0.005 * mid_price), 1.0) * self.confidence_decay

        trend_strength = self.price_momentum()
        trend_boost = min(abs(trend_strength) / (0.002 * mid_price), 1.0)
        level_proximity = self.proximity_to_level(mid_price)
        trend_volatility_boost = min(np.std(self.prev_mid_prices[-10:]) / (0.002 * mid_price), 1.0)

        base_lambda = self.get_session_lambda()
        herd_bias = ClientFlowSimulator.herd_sentiment * 30
        rumor_impact = ClientFlowSimulator.news_rumor_pressure * 100
        noise_boost = random.uniform(0.05, 0.1)

        action_probability = (
            base_lambda +
            0.3 * trend_boost +
            0.4 * level_proximity +
            0.3 * trend_volatility_boost +
            herd_bias +
            rumor_impact +
            noise_boost
        )
        action_probability = max(0.0, min(action_probability, 1.0))

        if self.tick_counter % 1000 == 0 and random.random() < 0.3:
            return [Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.uid,
                side=random.choice([OrderSide.BID, OrderSide.ASK]),
                order_type=OrderType.MARKET,
                volume=random.randint(1500, 3000),
                price=None
            )]

        if random.random() > action_probability:
            return []

        if self.behavior == "momentum":
            side = OrderSide.BID if signal_strength > 0 else OrderSide.ASK
        elif self.behavior == "mean_reversion":
            side = OrderSide.ASK if signal_strength > 0 else OrderSide.BID
        elif self.behavior == "panic":
            side = OrderSide.ASK if self.position > 0 else OrderSide.BID
        elif self.behavior == "greedy":
            side = OrderSide.BID if trend_strength > 0 else OrderSide.ASK
        else:
            side = OrderSide.BID if random.random() < 0.5 else OrderSide.ASK

        if random.random() > confidence * self.confidence_decay:
            return []

        volume_boost = 1.0 + 0.7 * trend_boost + 0.5 * level_proximity + abs(ClientFlowSimulator.herd_sentiment) * 2
        volatility_factor = 1.0 + min(np.std(self.prev_mid_prices[-10:]) / (0.001 * mid_price), 1.0)
        risk_factor = 1.0 - self.risk_aversion

        volume = (
            np.random.pareto(1.7) *
            20000 *
            random.uniform(0.8, 1.3) *
            volume_boost *
            volatility_factor *
            risk_factor
        )
        volume = max(2000, min(volume, 500_000))

        if self.confidence_decay < 1.0:
            self.confidence_decay = min(self.confidence_decay + self.recovery_rate, 1.0)
        elif self.confidence_decay > 1.0:
            self.confidence_decay = max(self.confidence_decay - self.recovery_rate, 1.0)

        return [Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.uid,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            price=None
        )]


# === Постоянный клиентский поток еще ===
class ClientFlowSimulatorv2:
    def __init__(self, uid, behavior=None):
        self.uid = uid
        self.position = 0.0
        self.capital = 25_000_000

        # === Поведенческая стратегия
        self.behavior = behavior or random.choices(
            ["momentum", "mean_reversion", "random"],
            weights=[0.4, 0.4, 0.2]
        )[0]

        # === Эмоциональное состояние
        self.mood = "neutral"  # может быть "greedy", "fearful", "neutral"

        # === Индикаторы тренда
        self.ema_short = None
        self.ema_long = None
        self.alpha_short = 0.3
        self.alpha_long = 0.05

        self.inertia_timer = random.randint(150, 300)

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.uid:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        elif trade['sell_agent'] == self.uid:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']

    def get_session_lambda(self):
        hour = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
        if 8 <= hour < 11:
            return 0.4
        elif 11 <= hour < 12:
            return 0.25
        elif 14.5 <= hour < 22:
            return 0.6
        elif 22 <= hour or hour < 6:
            return 0.6
        else:
            return 0.20

    def update_emas(self, mid_price):
        if self.ema_short is None:
            self.ema_short = mid_price
            self.ema_long = mid_price
        else:
            self.ema_short = self.alpha_short * mid_price + (1 - self.alpha_short) * self.ema_short
            self.ema_long = self.alpha_long * mid_price + (1 - self.alpha_long) * self.ema_long

    def update_mood(self, confidence):
        if confidence > 0.6:
            self.mood = "greedy"
        elif confidence < 0.2:
            self.mood = "fearful"
        else:
            self.mood = "neutral"

    def step(self, order_book, market_context=None):
        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []

        λ = self.get_session_lambda()
        if random.random() > λ:
            # клиент в любом случае торгует, но в неактивной сессии объём будет слабый
            base_volume = 0.25
        else:
            base_volume = 1.0

        mid_price = (best_bid + best_ask) / 2
        self.update_emas(mid_price)

        self.inertia_timer -= 1
        if self.inertia_timer <= 0:
            self.behavior = random.choices(
                ["momentum", "mean_reversion", "random"],
                weights=[0.4, 0.4, 0.2]
            )[0]
            self.inertia_timer = random.randint(150, 300)

        signal_strength = self.ema_short - self.ema_long if self.ema_short and self.ema_long else 0
        confidence = min(abs(signal_strength) / (0.005 * mid_price), 1.0)
        self.update_mood(confidence)

        # === Определение стороны заявки
        if self.behavior == "momentum":
            side = OrderSide.BID if signal_strength > 0 else OrderSide.ASK
        elif self.behavior == "mean_reversion":
            side = OrderSide.ASK if signal_strength > 0 else OrderSide.BID
        else:
            side = OrderSide.BID if random.random() < 0.5 else OrderSide.ASK

        # === Уровень агрессии на объём
        volume_multiplier = 1.0
        if self.mood == "greedy":
            volume_multiplier *= 1.5
        elif self.mood == "fearful":
            volume_multiplier *= 0.5

        if confidence < 0.2:
            volume_multiplier *= 0.3  # слабый сигнал → мелкий ордер
        elif confidence < 0.5:
            volume_multiplier *= 0.7

        base_size = np.random.pareto(1.7) * 20000 * random.uniform(0.7, 1.5)
        volume = max(2430, min(base_size * base_volume * volume_multiplier, 5_000_000))

        return [Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.uid,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            price=None
        )]

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        trade_pnl = fill_price * fill_size if side == OrderSide.ASK else -fill_price * fill_size
        self.capital += trade_pnl
        self.position += fill_size if side == OrderSide.BID else -fill_size

        if not hasattr(self, 'pnl_history'):
            self.pnl_history = deque(maxlen=30)
        self.pnl_history.append(trade_pnl)

        total_pnl = sum(self.pnl_history)
        if total_pnl < -500_000:
            self.behavior = "mean_reversion"  # переходит в защиту
        elif total_pnl > 1_000_000:
            self.behavior = "momentum"  # усиливает агрессию


# === Корпоративный поток (исполняет крупные заявки клиентов) ===
class CorporateFlowExecutor:
    def __init__(self, uid):
        self.uid = uid
        self.capital = 150_000_000
        self.in_progress = None
        self.cooldown = 0
        self.style = "TWAP"  # может быть "TWAP" или "POV"
        self.slice_timer = 0
        self.max_slippage = 0.01
        self.execution_history = []
        self.recent_fills = 0

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.uid:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        elif trade['sell_agent'] == self.uid:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']

    def estimate_liquidity(self, order_book, side):
        book = order_book.bids if side == OrderSide.BID else order_book.asks
        prices = sorted(book.keys(), reverse=(side == OrderSide.BID))
        total_volume = 0
        for price in prices[:3]:  # первые 3 уровня
            total_volume += book[price].total_volume()
        return total_volume

    def switch_execution_style(self, volatility):
        if volatility > 0.01:
            self.style = "POV"
        elif volatility < 0.003:
            self.style = "TWAP"
        # else: держим текущую

    def compute_volatility(self):
        if len(self.execution_history) < 10:
            return 0
        prices = self.execution_history[-10:]
        returns = [abs((b - a) / a) for a, b in zip(prices[:-1], prices[1:])]
        return np.std(returns)

    def step(self, order_book, ctx):
        orders = []
        if self.cooldown > 0:
            self.cooldown -= 1
            return orders

        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        volatility = self.compute_volatility()

        self.switch_execution_style(volatility)

        if spread / mid > 0.004:
            return orders

        if self.in_progress is None and random.random() < 0.02:
            side = OrderSide.BID if random.random() < 0.5 else OrderSide.ASK
            total_vol = random.randint(500_000, 1_500_000)
            slice_count = random.randint(5, 12)
            self.in_progress = {
                "side": side,
                "remaining": slice_count,
                "volume_per_slice": total_vol // slice_count,
                "target_mid": mid
            }

        if self.in_progress:
            task = self.in_progress
            if self.slice_timer > 0:
                self.slice_timer -= 1
                return orders

            side = task["side"]
            vol = task["volume_per_slice"]
            target_mid = task["target_mid"]

            slippage = abs(mid - target_mid) / target_mid
            if slippage > self.max_slippage:
                self.cooldown = 5
                return orders

            liquidity = self.estimate_liquidity(order_book, side)
            if self.style == "POV" and liquidity < vol:
                vol = int(liquidity * 0.7)

            use_market = (
                    (self.style == "POV" and random.random() < 0.3) or
                    (volatility > 0.01 and random.random() < 0.5)
            )

            price = None if use_market else (
                best_bid if side == OrderSide.BID else best_ask
            )
            order_type = OrderType.MARKET if use_market else OrderType.LIMIT

            orders.append(Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.uid,
                side=side,
                order_type=order_type,
                volume=vol,
                price=price
            ))

            task["remaining"] -= 1
            self.slice_timer = random.randint(3, 8)

            if task["remaining"] <= 0:
                self.in_progress = None
                self.cooldown = 20

        return orders

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        self.capital += (-fill_price if side == OrderSide.BID else fill_price) * fill_size
        self.execution_history.append(fill_price)
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)


# === Риск-менеджер банка ===
class BankRiskManager:
    def __init__(self, uid):
        self.uid = uid
        self.position = 0.0
        self.pnl = 0.0
        self.capital = 100_000_000
        self.limits = {
            "max_position": 25_000_000,
            "drawdown_limit": -10_000_000
        }

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.uid:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        elif trade['sell_agent'] == self.uid:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']

    def step(self, order_book, ctx):
        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []

        mid = (best_bid + best_ask) / 2
        unrealized = mid * self.position

        if abs(self.position) > self.limits["max_position"] or self.pnl + unrealized < self.limits["drawdown_limit"]:
            side = OrderSide.ASK if self.position > 0 else OrderSide.BID
            return [Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.uid,
                side=side,
                order_type=OrderType.MARKET,
                volume=abs(self.position),
                price=None
            )]
        return []

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        trade_pnl = fill_price * fill_size if side == OrderSide.ASK else -fill_price * fill_size
        self.position += fill_size if side == OrderSide.BID else -fill_size
        self.pnl += trade_pnl
        self.capital += trade_pnl


# === Казначейство банка ===
class TreasuryDesk:
    def __init__(self, uid):
        self.uid = uid
        self.net_position = 0.0
        self.threshold = 5_000_000
        self.position = 0.0
        self.capital = 100_000_000

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.uid:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        elif trade['sell_agent'] == self.uid:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']

    def update_net_position(self, all_positions):
        self.net_position = sum(all_positions)

    def step(self, order_book, ctx):
        if abs(self.net_position) < self.threshold:
            return []

        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []

        side = OrderSide.ASK if self.net_position > 0 else OrderSide.BID
        price = best_bid if side == OrderSide.ASK else best_ask
        volume = min(abs(self.net_position), 5_000_000)

        return [Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.uid,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            price=None
        )]


# === Алготрейдинг-деск банка ===
class QuantDesk:
    def __init__(self, uid):
        self.uid = uid
        self.position = 0.0
        self.capital = 50_000_000
        self.recent_prices = []
        self.window_size = 20

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.uid:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        elif trade['sell_agent'] == self.uid:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']

    def step(self, order_book, ctx):
        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []

        mid_price = (best_bid + best_ask) / 2
        self.recent_prices.append(mid_price)
        if len(self.recent_prices) > self.window_size:
            self.recent_prices.pop(0)

        if len(self.recent_prices) < self.window_size:
            return []

        sma = sum(self.recent_prices) / len(self.recent_prices)
        signal = mid_price - sma

        if abs(signal) < 0.03 * sma:
            return []

        side = OrderSide.BID if signal < 0 else OrderSide.ASK
        volume = 100_000 + random.randint(0, 150_000)

        return [Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.uid,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            price=None
        )]

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        pnl = (-fill_price if side == OrderSide.BID else fill_price) * fill_size
        self.capital += pnl
        self.position += fill_size if side == OrderSide.BID else -fill_size


# === маркетос банка ===
class HighFreqMarketMaker:
    def __init__(self, uid):
        self.uid = uid
        self.position = 0.0
        self.capital = 500_000_000
        self.order_volume = 50_000

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.uid:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
        elif trade['sell_agent'] == self.uid:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']

    def step(self, order_book, market_context=None):
        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []

        mid = (best_bid + best_ask) / 2

        bid_price = quant(mid - TICK / 2)
        ask_price = quant(mid + TICK / 2)

        return [
            Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.uid,
                side=OrderSide.BID,
                order_type=OrderType.LIMIT,
                volume=self.order_volume,
                price=bid_price
            ),
            Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.uid,
                side=OrderSide.ASK,
                order_type=OrderType.LIMIT,
                volume=self.order_volume,
                price=ask_price
            )
        ]

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        pnl = (-fill_price if side == OrderSide.BID else fill_price) * fill_size
        self.capital += pnl
        self.position += fill_size if side == OrderSide.BID else -fill_size


# ── Инициатор импульсов ─────────────────────────
class InitiatorUnit:
    def __init__(self, agent_id, capital):
        self.agent_id = agent_id
        self.capital = capital
        self.position = 0
        self.entry_price = None
        self.pnl = 0.0

        self.signal = None  # 'long' или 'short' или None
        self.active = False

        self.order_cooldown = 0
        self.max_order_size = capital * 0.01
        self.accumulated_size = 0
        self.state = "idle"  # idle / accumulating / holding / distributing

        self.lookback_trades = 500  # например, 500 последних сделок
        self.mode = "trend_driver"

    def compute_metrics(self, order_book: OrderBook):
        # Возьмем последние трейды
        trades = list(order_book.trade_history)[-self.lookback_trades:]

        if len(trades) < self.lookback_trades:
            return None

        prices = [t['price'] for t in trades]
        volumes = [t['volume'] for t in trades]
        buy_volumes = sum(t['volume'] for t in trades if t['buy_agent'] == self.agent_id)
        sell_volumes = sum(t['volume'] for t in trades if t['sell_agent'] == self.agent_id)

        # EMA - упрощенно среднее по последним ценам (можно заменить на классические EMA)
        ema_fast = sum(prices[-100:]) / 100
        ema_slow = sum(prices[-400:]) / 400

        delta = buy_volumes - sell_volumes

        impulse = (prices[-1] - prices[-50]) / prices[-50]

        current_price = prices[-1]

        # объемы с книги ордеров, например топ 5 уровней по BID и ASK
        bids = order_book.get_order_book_snapshot(depth=5)['bids']
        asks = order_book.get_order_book_snapshot(depth=5)['asks']
        bid_volume = sum(level['volume'] for level in bids)
        ask_volume = sum(level['volume'] for level in asks)

        imbalance = bid_volume - ask_volume

        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "delta": delta,
            "impulse": impulse,
            "imbalance": imbalance,
            "current_price": current_price
        }

    def generate_orders(self, order_book: OrderBook):
        metrics = self.compute_metrics(order_book)
        if metrics is None:
            return []

        orders = []

        if not self.active:
            if (
                    metrics["ema_fast"] > metrics["ema_slow"] * 1.001 and
                    metrics["delta"] > 10000 and
                    metrics["impulse"] > 0.002 and
                    metrics["imbalance"] > 1000
            ):
                self.signal = "long"
                self.active = True
                self.state = "accumulating"
                self.entry_price = metrics["current_price"]
            elif (
                    metrics["ema_fast"] < metrics["ema_slow"] * 0.999 and
                    metrics["delta"] < -10000 and
                    metrics["impulse"] < -0.002 and
                    metrics["imbalance"] < -1000
            ):
                self.signal = "short"
                self.active = True
                self.state = "accumulating"
                self.entry_price = metrics["current_price"]

        if self.active:
            price = metrics["current_price"]

            if self.state == "accumulating":
                if self.accumulated_size < self.capital * 0.6:
                    size = random.uniform(0.1, 0.3) * self.max_order_size
                    side = OrderSide.BID if self.signal == "long" else OrderSide.ASK
                    orders.append(Order(
                        order_id=str(uuid.uuid4()),
                        agent_id=self.agent_id,
                        side=side,
                        order_type=OrderType.MARKET,
                        volume=size,
                        price=None
                    ))

                    self.accumulated_size += size
                else:
                    self.state = "holding"

            elif self.state == "holding":
                if self.signal == "long" and price < self.entry_price * 0.997:
                    self.state = "exit"
                elif self.signal == "short" and price > self.entry_price * 1.003:
                    self.state = "exit"

                rr = 2.0
                if self.signal == "long" and price > self.entry_price * (1 + 0.01 * rr):
                    self.state = "distributing"
                elif self.signal == "short" and price < self.entry_price * (1 - 0.01 * rr):
                    self.state = "distributing"

            elif self.state == "distributing":
                if self.accumulated_size > 0:
                    size = random.uniform(0.1, 0.3) * self.max_order_size
                    side = OrderSide.ASK if self.signal == "long" else OrderSide.BID
                    orders.append(Order(
                        order_id=str(uuid.uuid4()),
                        agent_id=self.agent_id,
                        side=side,
                        order_type=OrderType.MARKET,
                        volume=size,
                        price=None
                    ))

                    self.accumulated_size -= size
                else:
                    self.reset()


            elif self.state == "exit":

                if self.accumulated_size > 0:
                    side = OrderSide.ASK if self.signal == "long" else OrderSide.BID
                    orders.append(Order(
                        order_id=str(uuid.uuid4()),
                        agent_id=self.agent_id,
                        side=side,
                        order_type=OrderType.MARKET,
                        volume=self.accumulated_size,
                        price=None
                    ))
                    self.accumulated_size = 0
                self.reset()

        return orders

    def reset(self):
        self.active = False
        self.signal = None
        self.accumulated_size = 0
        self.entry_price = None
        self.state = "idle"


# === Менеджер банка v2 ===
class BankAgentManagerv2:
    def __init__(self, agent_id, total_capital):
        self.agent_id = agent_id
        self.total_capital = total_capital
        self.capital = total_capital

        self.clients = [ClientFlowSimulator(f"{agent_id}_client_{i}") for i in range(8)]
        self.clients_v2 = [ClientFlowSimulatorv2(f"{agent_id}_client_{i}") for i in range(8)]
        self.corporate_flows = [CorporateFlowExecutor(f"{agent_id}_corp_{i}") for i in range(3)]
        self.risk_units = [BankRiskManager(f"{agent_id}_risk_{i}") for i in range(1)]
        self.treasury = TreasuryDesk(f"{agent_id}_treasury")
        self.quant = QuantDesk(f"{agent_id}_quant")
        self.initiator = InitiatorUnit(agent_id=self.agent_id, capital=self.capital)

        self.step_counter = 0
        self.volume_this_minute = 0
        self.log_buffer = deque(maxlen=60)  # храним 60 минут

    def on_trade(self, trade: dict):
        if trade['buy_agent'] == self.agent_id:
            self.position += trade['volume']
            self.capital -= trade['price'] * trade['volume']
            self.initiator.position += trade['volume']
            self.initiator.capital = self.capital

        elif trade['sell_agent'] == self.agent_id:
            self.position -= trade['volume']
            self.capital += trade['price'] * trade['volume']
            self.initiator.position -= trade['volume']
            self.initiator.capital = self.capital

    def generate_orders(self, order_book, market_context=None):
        orders = []

        all_positions = []

        for client in self.clients:
            orders_c = client.step(order_book, market_context)
            orders.extend(orders_c)
            all_positions.append(client.position)
            self.volume_this_minute += sum(o.volume for o in orders_c)

        for client in self.clients_v2:
            orders_c = client.step(order_book, market_context)
            orders.extend(orders_c)
            all_positions.append(client.position)
            self.volume_this_minute += sum(o.volume for o in orders_c)

        for corp in self.corporate_flows:
            orders_c = corp.step(order_book, market_context)
            orders.extend(orders_c)
            self.volume_this_minute += sum(o.volume for o in orders_c)

        for risk in self.risk_units:
            orders_c = risk.step(order_book, market_context)
            orders.extend(orders_c)
            all_positions.append(risk.position)
            self.volume_this_minute += sum(o.volume for o in orders_c)

        self.treasury.update_net_position(all_positions)
        orders_c = self.treasury.step(order_book, market_context)
        orders.extend(orders_c)
        self.volume_this_minute += sum(o.volume for o in orders_c)

        orders_c = self.quant.step(order_book, market_context)
        orders.extend(orders_c)
        self.volume_this_minute += sum(o.volume for o in orders_c)

        orders_c = self.initiator.generate_orders(order_book)
        orders.extend(orders_c)
        self.volume_this_minute += sum(o.volume for o in orders_c)

        self.step_counter += 1
        if self.step_counter % 60 == 0:  # допустим, 1 шаг = 1 секунда симуляции
            self.log_bank_status()

        return orders

    def log_bank_status(self):
        total_cap = sum(agent.capital for agent in (
                self.clients + self.clients_v2 + self.corporate_flows +
                self.risk_units + [self.quant]
        ))

        total_position = sum(
            agent.position for agent in (
                    self.clients + self.clients_v2 + self.risk_units + [self.quant]
            )
        )

        log_entry = {
            "step": self.step_counter,
            "total_capital": round(total_cap, 2),
            "net_position": round(total_position, 2),
            "volume_last_minute": round(self.volume_this_minute, 2)
        }

        self.log_buffer.append(log_entry)

        bank_logger.info(
            f"Step {log_entry['step']}, "
            f"Capital={log_entry['total_capital']:,.2f}, "
            f"Position={log_entry['net_position']:,.2f}, "
            f"1-min Volume={log_entry['volume_last_minute']:,.2f}"
        )

        self.volume_this_minute = 0

    def on_order_filled(self, order_id, fill_price, fill_size, side, agent_uid):
        for agent_list in [self.clients, self.clients_v2, self.corporate_flows, self.risk_units, [self.quant]]:
            for agent in agent_list:
                if agent.uid == agent_uid:
                    agent.on_order_filled(order_id, fill_price, fill_size, side)
                    return


