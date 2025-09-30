import heapq 
import time
import random
from collections import deque
from typing import Dict, Optional, List, Any

from order import Order, OrderSide, OrderType
from order import quant
 
class PriceLevel:
    def __init__(self, price: float):
        self.price = price
        self.orders: deque[Order] = deque()

    def add_order(self, order: Order):
        self.orders.append(order)

    def remove_filled_or_inactive_orders(self):
        while self.orders and not self.orders[0].is_active():
            self.orders.popleft()

    def total_volume(self) -> float:
        return sum(order.remaining_volume() for order in self.orders if order.is_active())

    def __bool__(self):
        self.remove_filled_or_inactive_orders()
        return len(self.orders) > 0

    def __repr__(self):
        return f"PriceLevel(price={self.price}, orders={len(self.orders)})"


class OrderBook:
    def __init__(self):
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}
        self.bid_prices: List[float] = []
        self.ask_prices: List[float] = []
        self.orders: Dict[str, Order] = {}
        self.last_trade_price: Optional[float] = None
        self.trade_history: deque[Dict[str, Any]] = deque(maxlen=1000)

    def _add_price_level(self, side: OrderSide, price: float):
        if side == OrderSide.BID:
            if price not in self.bids:
                self.bids[price] = PriceLevel(price)
                heapq.heappush(self.bid_prices, -price)
        else:
            if price not in self.asks:
                self.asks[price] = PriceLevel(price)
                heapq.heappush(self.ask_prices, price)

    def _remove_price_level_if_empty(self, side: OrderSide, price: float):
        if side == OrderSide.BID:
            pl = self.bids.get(price)
            if pl is None or not pl:
                if price in self.bids:
                    del self.bids[price]
                if -price in self.bid_prices:
                    self.bid_prices.remove(-price)
                    heapq.heapify(self.bid_prices)
        else:
            pl = self.asks.get(price)
            if pl is None or not pl:
                if price in self.asks:
                    del self.asks[price]
                if price in self.ask_prices:
                    self.ask_prices.remove(price)
                    heapq.heapify(self.ask_prices)

    def _best_bid_price(self) -> Optional[float]:
        while self.bid_prices:
            price = -self.bid_prices[0]
            if price in self.bids and self.bids[price]:
                return price
            else:
                heapq.heappop(self.bid_prices)
        return None

    def _best_ask_price(self) -> Optional[float]:
        while self.ask_prices:
            price = self.ask_prices[0]
            if price in self.asks and self.asks[price]:
                return price
            else:
                heapq.heappop(self.ask_prices)
        return None

    def add_order(self, order: Order) -> List[Dict[str, Any]]:
        self.orders[order.order_id] = order
        if order.order_type == OrderType.LIMIT:
            if order.price is None:
                raise ValueError("Limit order must have a price")
            order.price = quant(order.price)
            self._add_price_level(order.side, order.price)
            book = self.bids if order.side == OrderSide.BID else self.asks
            book[order.price].add_order(order)
            return []
        else:
            return self._match_market_order(order)


    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order and order.is_active():
            order.cancel()
            return True
        return False

    def _match_market_order(self, market_order: Order) -> List[Dict[str, Any]]:
        trades = []

        if market_order.side == OrderSide.BID:
            opposite_book = self.asks
            price_getter = self._best_ask_price
        else:
            opposite_book = self.bids
            price_getter = self._best_bid_price

        slippage_factor = 0.005
        trades_for_slippage_calc = []

        while market_order.is_active():
            best_price = price_getter()
            if best_price is None:
                break

            level = opposite_book[best_price]
            level.remove_filled_or_inactive_orders()
            if not level:
                self._remove_price_level_if_empty(
                    OrderSide.ASK if market_order.side == OrderSide.BID else OrderSide.BID,
                    best_price
                )
                continue

            resting_order = level.orders[0]
            trade_qty = min(market_order.remaining_volume(), resting_order.remaining_volume())

            market_order.fill(trade_qty)
            resting_order.fill(trade_qty)

            trades_for_slippage_calc.append((trade_qty, best_price))

            timestamp = time.time()
            trade = {
                'price': best_price,
                'volume': trade_qty,
                'buy_order_id': market_order.order_id if market_order.side == OrderSide.BID else resting_order.order_id,
                'sell_order_id': resting_order.order_id if market_order.side == OrderSide.BID else market_order.order_id,
                'buy_agent': market_order.agent_id if market_order.side == OrderSide.BID else resting_order.agent_id,
                'sell_agent': resting_order.agent_id if market_order.side == OrderSide.BID else market_order.agent_id,
                'timestamp': timestamp,
                'taker_side': 'buy' if market_order.side == OrderSide.BID else 'sell'
            }
            trades.append(trade)
            self.trade_history.append(trade)

            if not resting_order.is_active():
                level.remove_filled_or_inactive_orders()
                self._remove_price_level_if_empty(
                    OrderSide.ASK if market_order.side == OrderSide.BID else OrderSide.BID,
                    best_price
                )

        if trades_for_slippage_calc:
            traded_value = sum(q * p for q, p in trades_for_slippage_calc)
            total_qty = sum(q for q, _ in trades_for_slippage_calc)
            self.last_trade_price = traded_value / total_qty

        if market_order.remaining_volume() > 0:
            market_order.cancel()

        return trades




    def match(self) -> List[Dict[str, Any]]:
        """
        Основной матчмейкер лимитных ордеров. Пока лучшая BID >= лучшей ASK, совершаем сделки:
          – выбираем пару (_bid_level_, _ask_level_), находим минимальный объём,
            заполняем его в обе стороны, записываем сделку.
          – удаляем уровни, если на них больше нет активных ордеров.
        Возвращает список совершённых сделок (каждая — словарь).
        """
        trades: List[Dict[str, Any]] = []

        while True:
            best_bid = self._best_bid_price()
            best_ask = self._best_ask_price()
            if best_bid is None or best_ask is None or best_bid < best_ask:
                break

            bid_level = self.bids[best_bid]
            ask_level = self.asks[best_ask]

            bid_level.remove_filled_or_inactive_orders()
            ask_level.remove_filled_or_inactive_orders()

            if not bid_level or not ask_level:
                # Если какой-то из уровней стал пустым, пересобираем кучу
                if not bid_level:
                    self._remove_price_level_if_empty(OrderSide.BID, best_bid)
                if not ask_level:
                    self._remove_price_level_if_empty(OrderSide.ASK, best_ask)
                continue

            bid_order = bid_level.orders[0]
            ask_order = ask_level.orders[0]

            # Цена сделки — та, чей ордер выставлен раньше (time priority)
            trade_price = ask_order.price if ask_order.timestamp < bid_order.timestamp else bid_order.price
            trade_qty = min(bid_order.remaining_volume(), ask_order.remaining_volume())

            bid_order.fill(trade_qty)
            ask_order.fill(trade_qty)

            self.last_trade_price = trade_price
            timestamp = time.time()
            taker_is_bid = (bid_order.timestamp > ask_order.timestamp)

            trade = {
                'price': trade_price,
                'volume': trade_qty,
                'buy_order_id': bid_order.order_id,
                'sell_order_id': ask_order.order_id,
                'buy_agent': bid_order.agent_id,
                'sell_agent': ask_order.agent_id,
                'timestamp': timestamp,
                'taker_side': 'buy' if taker_is_bid else 'sell'
            }
            trades.append(trade)
            self.trade_history.append(trade)

            # Если BID-ордер полностью исполнен, удаляем его из уровня
            if not bid_order.is_active():
                bid_level.remove_filled_or_inactive_orders()
                self._remove_price_level_if_empty(OrderSide.BID, best_bid)

            # Если ASK-ордер полностью исполнен, удаляем его из уровня
            if not ask_order.is_active():
                ask_level.remove_filled_or_inactive_orders()
                self._remove_price_level_if_empty(OrderSide.ASK, best_ask)

        return trades

    def tick(self):
        """
        Должно вызываться каждый «тик» (часто). Уменьшаем TTL у всех ордеров
        и отменяем просроченные (TTL ≤ 0). Затем чистим пустые уровни.
        """
        to_cancel = []
        for oid, order in list(self.orders.items()):
            if order.ttl is not None and order.is_active():
                order.ttl -= 1
                if order.ttl <= 0:
                    to_cancel.append(oid)

        for oid in to_cancel:
            self.cancel_order(oid)

        # После этого проходим по каждой цене и удаляем пустые уровни
        for price in list(self.bids.keys()):
            self._remove_price_level_if_empty(OrderSide.BID, price)
        for price in list(self.asks.keys()):
            self._remove_price_level_if_empty(OrderSide.ASK, price)

    def get_order_book_snapshot(self, depth: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Возвращает топ‐depth уровней книжки:
            {'bids': [ {'price': .., 'volume': ..}, … ],
             'asks': [ {'price': .., 'volume': ..}, … ] }
        """
        bids_snapshot: List[Dict[str, Any]] = []
        asks_snapshot: List[Dict[str, Any]] = []

        bid_prices_sorted = sorted(self.bids.keys(), reverse=True)[:depth]
        for price in bid_prices_sorted:
            level = self.bids[price]
            vol = level.total_volume()
            if vol > 0:
                bids_snapshot.append({
                    "price": level.price,
                    "volume": level.total_volume(),
                    "source": level.orders[0].source if level.orders and hasattr(level.orders[0], 'source') else None
                })

        ask_prices_sorted = sorted(self.asks.keys())[:depth]
        for price in ask_prices_sorted:
            level = self.asks[price]
            vol = level.total_volume()
            if vol > 0:
                asks_snapshot.append({
                    'price': price,
                    'volume': vol,
                    'source': level.orders[0].source if level.orders and hasattr(level.orders[0], 'source') else None
                })

        return {'bids': bids_snapshot, 'asks': asks_snapshot}

    def __repr__(self):
        return (f"OrderBook(bids={len(self.bids)}, asks={len(self.asks)}, "
                f"orders={len(self.orders)}, last_trade_price={self.last_trade_price})")
