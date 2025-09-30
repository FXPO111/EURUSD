import heapq
import time
from collections import deque
from typing import Optional, List, Dict, Any
import logging

from order import Order, OrderSide, OrderType

# --- Логирование ---
logging.basicConfig(
    filename='matching_engine.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class PriceLevel:
    """
    Структура для ордеров одного уровня цены.
    Хранит ордера FIFO в deque.
    """
    def __init__(self, price: float):
        self.price = price
        self.orders: deque[Order] = deque()

    def add_order(self, order: Order):
        self.orders.append(order)
        logging.debug(f"Added order {order.order_id} to price level {self.price}")

    def remove_filled_or_inactive_orders(self):
        while self.orders and not self.orders[0].is_active():
            removed = self.orders.popleft()
            logging.debug(f"Removed inactive/filled order {removed.order_id} from price level {self.price}")

    def total_volume(self) -> float:
        self.remove_filled_or_inactive_orders()
        return sum(o.remaining_volume() for o in self.orders if o.is_active())

    def is_empty(self) -> bool:
        self.remove_filled_or_inactive_orders()
        return len(self.orders) == 0

    def __bool__(self):
        return not self.is_empty()

    def __repr__(self):
        return f"PriceLevel(price={self.price}, orders_count={len(self.orders)})"

class OrderBook:
    """
    Хранит и управляет стаканом и ордерами.
    """

    def __init__(self):
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}

        self.bid_prices: List[float] = []  # max-heap (через отрицательные цены)
        self.ask_prices: List[float] = []  # min-heap

        self.orders: Dict[str, Order] = {}

    def _add_price_level(self, side: OrderSide, price: float):
        book = self.bids if side == OrderSide.BID else self.asks
        prices_heap = self.bid_prices if side == OrderSide.BID else self.ask_prices

        if price not in book:
            book[price] = PriceLevel(price)
            if side == OrderSide.BID:
                heapq.heappush(prices_heap, -price)
            else:
                heapq.heappush(prices_heap, price)
            logging.debug(f"Added price level {price} for {side.name}")

    def _remove_price_level_if_empty(self, side: OrderSide, price: float):
        book = self.bids if side == OrderSide.BID else self.asks
        prices_heap = self.bid_prices if side == OrderSide.BID else self.ask_prices

        level = book.get(price)
        if level and level.is_empty():
            del book[price]
            # Удаляем цену из кучи
            if side == OrderSide.BID:
                prices_heap.remove(-price)
            else:
                prices_heap.remove(price)
            heapq.heapify(prices_heap)
            logging.debug(f"Removed empty price level {price} for {side.name}")

    def add_order(self, order: Order):
        """
        Добавляет ордер в книгу (не матчинг!)
        """
        if order.order_id in self.orders:
            logging.warning(f"Order {order.order_id} уже в книге")
            return

        side = order.side
        self._add_price_level(side, order.price)
        level = self.bids[order.price] if side == OrderSide.BID else self.asks[order.price]
        level.add_order(order)
        self.orders[order.order_id] = order
        logging.debug(f"Order {order.order_id} added to book at price {order.price}")

    def remove_order(self, order_id: str):
        """
        Отменяет и удаляет ордер из книги.
        """
        order = self.orders.get(order_id)
        if not order:
            logging.warning(f"Order {order_id} не найден для удаления")
            return False

        order.cancel()
        # Удаление из PriceLevel происходит при очистке (lazy removal)
        return True

    def best_bid(self) -> Optional[float]:
        while self.bid_prices:
            price = -self.bid_prices[0]
            if price in self.bids and self.bids[price]:
                return price
            heapq.heappop(self.bid_prices)
        return None

    def best_ask(self) -> Optional[float]:
        while self.ask_prices:
            price = self.ask_prices[0]
            if price in self.asks and self.asks[price]:
                return price
            heapq.heappop(self.ask_prices)
        return None

    def get_price_level(self, side: OrderSide, price: float) -> Optional[PriceLevel]:
        book = self.bids if side == OrderSide.BID else self.asks
        return book.get(price)

    def clean_empty_levels(self):
        for side in [OrderSide.BID, OrderSide.ASK]:
            book = self.bids if side == OrderSide.BID else self.asks
            for price in list(book.keys()):
                self._remove_price_level_if_empty(side, price)

    def remove_filled_orders(self):
        # Удаляем из self.orders неактивные ордера
        for oid in list(self.orders):
            if not self.orders[oid].is_active():
                del self.orders[oid]

    def __repr__(self):
        return f"OrderBook(bids={list(self.bids.keys())}, asks={list(self.asks.keys())})"

class MatchingEngine:
    def __init__(self):
        self.order_book = OrderBook()
        self.last_trade_price: Optional[float] = None
        self.trade_history: deque[Dict[str, Any]] = deque(maxlen=1000)

    def _match_orders(self, taker_order: Order) -> List[Dict[str, Any]]:
        """
        Основной матчинг лимитных и рыночных ордеров.
        """
        trades = []
        side = taker_order.side
        opposite_side = OrderSide.ASK if side == OrderSide.BID else OrderSide.BID
        book = self.order_book.asks if side == OrderSide.BID else self.order_book.bids

        if taker_order.order_type == OrderType.MARKET and not book:
            logging.warning(
                f"Нет ликвидности для исполнения {side.name} маркет-ордера {taker_order.order_id}. Пропускаем.")
            return trades

        def get_best_price():
            return self.order_book.best_ask() if side == OrderSide.BID else self.order_book.best_bid()

        while taker_order.remaining_volume() > 0:
            best_price = get_best_price()
            if best_price is None:
                break

            # Проверка цены для лимитных ордеров
            if taker_order.order_type == OrderType.LIMIT:
                if side == OrderSide.BID and best_price > taker_order.price:
                    break
                if side == OrderSide.ASK and best_price < taker_order.price:
                    break

            level = book.get(best_price)
            if not level or level.is_empty():
                self.order_book._remove_price_level_if_empty(opposite_side, best_price)
                continue

            level.remove_filled_or_inactive_orders()
            for maker_order in list(level.orders):
                if not maker_order.is_active():
                    continue

                match_qty = min(taker_order.remaining_volume(), maker_order.remaining_volume())
                trade_price = maker_order.price

                maker_order.fill(match_qty)
                taker_order.fill(match_qty)

                trade = {
                    'price': trade_price,
                    'volume': match_qty,
                    'buy_order_id': taker_order.order_id if side == OrderSide.BID else maker_order.order_id,
                    'sell_order_id': maker_order.order_id if side == OrderSide.BID else taker_order.order_id,
                    'buy_agent': taker_order.agent_id if side == OrderSide.BID else maker_order.agent_id,
                    'sell_agent': maker_order.agent_id if side == OrderSide.BID else taker_order.agent_id,
                    'timestamp': time.time(),
                    'taker_side': 'buy' if side == OrderSide.BID else 'sell'
                }
                trades.append(trade)
                self.last_trade_price = trade_price

                logging.info(f"Trade executed: {trade}")

                if taker_order.remaining_volume() == 0:
                    break

            level.remove_filled_or_inactive_orders()
            if level.is_empty():
                self.order_book._remove_price_level_if_empty(opposite_side, best_price)

            if taker_order.remaining_volume() == 0:
                break

        return trades

    def add_order(self, order: Order):
        """
        Добавляем ордер в книгу и запускаем матчинг.
        """
        logging.info(f"Adding order {order.order_id} side={order.side.name} type={order.order_type.name} qty={order.volume}")

        if order.order_type == OrderType.MARKET:
            trades = self._match_orders(order)
            if order.remaining_volume() > 0:
                logging.warning(f"Market order {order.order_id} partially filled, unfilled qty: {order.remaining_volume()}")
            self.trade_history.extend(trades)
            return trades
        else:
            # Сначала матчинг по возможности
            trades = self._match_orders(order)
            self.trade_history.extend(trades)

            # Если остался незаполненный объём — кладём в книгу
            if order.remaining_volume() > 0:
                self.order_book.add_order(order)
                logging.info(f"Order {order.order_id} placed in book with remaining qty {order.remaining_volume()}")
            else:
                logging.info(f"Order {order.order_id} полностью исполнен при добавлении")
            return trades

    def cancel_order(self, order_id: str):
        """
        Отмена ордера.
        """
        success = self.order_book.remove_order(order_id)
        if success:
            logging.info(f"Order {order_id} cancelled")
        else:
            logging.warning(f"Order {order_id} cancel failed")

    def tick(self):
        """
        Вызов периодического обновления:
         - уменьшаем TTL
         - отменяем просроченные
         - чистим книгу
         - запускаем матчинг
        """
        to_cancel = []
        for order in self.order_book.orders.values():
            if order.ttl is not None and order.is_active():
                order.tick_ttl()
                if order.ttl == 0:
                    to_cancel.append(order.order_id)

        for oid in to_cancel:
            self.cancel_order(oid)

        self.order_book.remove_filled_orders()
        self.order_book.clean_empty_levels()

        # Можно вызывать матчинг повторно, если нужно
        # Например, если отмены открыли возможность для новых сделок

    def get_order_book_snapshot(self, depth: int = 10) -> dict:
        bids = []
        asks = []

        for price in sorted(self.order_book.bids.keys(), reverse=True)[:depth]:
            level = self.order_book.bids[price]
            vol = level.total_volume()
            if vol > 0:
                bids.append({'price': price, 'volume': vol})

        for price in sorted(self.order_book.asks.keys())[:depth]:
            level = self.order_book.asks[price]
            vol = level.total_volume()
            if vol > 0:
                asks.append({'price': price, 'volume': vol})

        return {'bids': bids, 'asks': asks}

