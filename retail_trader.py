# Файл: noise_trader.py

import uuid
import random
import numpy as np
from order import Order, OrderSide, OrderType
from collections import deque



class NoiseSubAgent:
    def __init__(self, agent_id, capital, style="random"):
        self.agent_id = agent_id
        self.capital = capital
        self.style = style

        # Состояние
        self.position = 0.0
        self.entry_price = None
        self.hold_steps = 0
        self.cooldown = 0

        # Психология
        self.fatigue = np.random.uniform(0.0, 0.3)
        self.mood = np.random.uniform(0.7, 1.3)
        self.tilt = 0.0

        # Частота действий
        self.activity_rate = np.random.uniform(0.1, 0.4)
        self.reaction_delay = random.randint(1, 5)

        # Память для стилей
        self.price_window = 5
        self.last_prices = []

    @staticmethod
    def _safe_best_price(book, name, fallback):
        attr = getattr(book, name, None)
        if attr is None:
            return fallback()
        if callable(attr):
            return attr()
        return attr

    def decide(self, order_book, current_step):
        last_price = order_book.last_trade_price
        if last_price is None:
            return None

        self.last_prices.append(last_price)
        if len(self.last_prices) > self.price_window:
            self.last_prices.pop(0)

        # фильтры усталости, кулдауна и т.д.
        if random.random() < self.fatigue:
            return None
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if current_step % self.reaction_delay != 0:
            return None
        if random.random() > self.activity_rate + self.tilt:
            return None

        base_size = max(1, self.capital * 10 / last_price)
        size = int(np.random.lognormal(mean=np.log(base_size * self.mood), sigma=0.5))
        if size <= 0:
            return None

        # Определяем сторону в зависимости от стиля
        if self.style == "random":
            side = random.choice([OrderSide.BID, OrderSide.ASK])
        elif self.style == "impulse" and len(self.last_prices) >= 2:
            delta = self.last_prices[-1] - self.last_prices[0]
            side = OrderSide.BID if delta > 0 else OrderSide.ASK
        elif self.style == "contrarian" and len(self.last_prices) >= 2:
            delta = self.last_prices[-1] - self.last_prices[0]
            side = OrderSide.ASK if delta > 0 else OrderSide.BID
        else:
            # "liquidity" или нет данных для impulse/contrarian
            side = random.choice([OrderSide.BID, OrderSide.ASK])

        # Всегда создаём рыночный ордер
        self.cooldown = random.randint(1, 3)
        return Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=side,
            volume=size,
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )


class RetailTrader:
    def __init__(self, agent_id, capital, num_subagents=50):
        self.agent_id = agent_id
        self.capital = capital
        self.subagents = []

        styles = ["random", "liquidity", "impulse", "contrarian"]
        for i in range(num_subagents):
            style = random.choice(styles)
            self.subagents.append(
                NoiseSubAgent(f"{agent_id}_sub_{i}", capital / num_subagents, style)
            )

        self.price_history = deque(maxlen=200)
        self.current_step = 0

    def generate_orders(self, order_book, market_context=None):
        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        self.price_history.append(last_price)
        self.current_step += 1

        orders = []
        for sub in self.subagents:
            order = sub.decide(order_book, self.current_step)
            if order:
                orders.append(order)

        return orders
