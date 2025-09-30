import uuid
import random
import time

from order import Order, OrderSide, OrderType


class PassiveDepthProvider:
    """
    Барьерная глубина с rolling TTL:
    - mid-зона очищается (±0.05).
    - Глубина начинается с ±0.06 и до ±2.00 (200 уровней).
    - На каждом уровне целевой объём детерминированно 100k–1M.
    - Если ордер истёк, сетка тут же восполняется до целевого.
    """

    def __init__(self, agent_id, capital, min_depth_percent=0.01, max_depth_percent=0.01, order_lifetime=10.0):
        self.agent_id = agent_id
        self.capital = capital
        self.min_depth_percent = min_depth_percent
        self.max_depth_percent = max_depth_percent
        self.order_lifetime = order_lifetime  # не используется напрямую, оставлено для совместимости

        self.levels = 200             # уровни покрытия
        self.start_i = 6              # глубина от ±0.06, mid-зона пустая
        self.mid_radius = 0.05        # зона чистки вокруг mid
        self.place_chunk = 200_000    # максимальный объём одного ордера при дозаливке

        self.all_active_orders = []
        self.placed_orders_by_price = {}
        self.existing_liquidity = {'bid': {}, 'ask': {}}

        # бутстрап
        self.bootstrap_active = False
        self.bootstrap_deadline = 0.0
        self.bootstrap_queue = []

        self.position = 0.0

    # ---------------- Bootstrapping ----------------

    def begin_bootstrap(self, saved_liq: dict, seconds: float = 15.0):
        self.bootstrap_queue.clear()

        def _norm(side):
            out = []
            for px, vol in (side or {}).items():
                try:
                    p2 = round(float(px), 2)
                    v2 = float(vol)
                    if v2 > 0:
                        out.append((p2, v2))
                except Exception:
                    continue
            return out

        bids = sorted(_norm(saved_liq.get('bids')), key=lambda x: -x[0])
        asks = sorted(_norm(saved_liq.get('asks')), key=lambda x: x[0])

        for p2, v2 in bids:
            self.bootstrap_queue.append(('bid', p2, v2))
            self.existing_liquidity['bid'][p2] = v2
        for p2, v2 in asks:
            self.bootstrap_queue.append(('ask', p2, v2))
            self.existing_liquidity['ask'][p2] = v2

        self.bootstrap_active = True
        self.bootstrap_deadline = time.time() + float(seconds)

    # ---------------- Cancel logic ----------------

    def cancel_expired_orders(self, order_book=None):
        """Rolling TTL: отменяем только те, чей индивидуальный ttl истёк. Остальные не трогаем."""
        now = time.time()
        expired = []
        for o in list(self.all_active_orders):
            ttl = getattr(o, 'ttl', None)  # None -> бессрочно
            if ttl is None:
                continue
            # небольшой детерминированный джиттер, чтобы не истекали одновременно
            jitter = (hash(o.order_id) & 0xffff) / 65535.0 * 0.2 * ttl
            if now - o.timestamp >= ttl + jitter:
                expired.append(o)

        if order_book and expired:
            for o in expired:
                try:
                    order_book.cancel_order(o.order_id)
                except Exception:
                    pass

        if expired:
            ids = {o.order_id for o in expired}
            self.all_active_orders = [o for o in self.all_active_orders if o.order_id not in ids]
            for px, oid in list(self.placed_orders_by_price.items()):
                if oid in ids:
                    del self.placed_orders_by_price[px]

    # ---------------- Mid cleaner ----------------

    def _cancel_mid_zone(self, order_book, mid, radius):
        if order_book is None:
            return
        to_cancel = []
        for price, level in list(order_book.bids.items()):
            if abs(price - mid) <= radius:
                for o in list(level.orders):
                    if o.agent_id == self.agent_id and o.is_active():
                        to_cancel.append(o.order_id)
        for price, level in list(order_book.asks.items()):
            if abs(price - mid) <= radius:
                for o in list(level.orders):
                    if o.agent_id == self.agent_id and o.is_active():
                        to_cancel.append(o.order_id)

        for oid in to_cancel:
            try:
                order_book.cancel_order(oid)
            except Exception:
                pass

        if to_cancel:
            ids = set(to_cancel)
            self.all_active_orders = [o for o in self.all_active_orders if o.order_id not in ids]
            for px, oid in list(self.placed_orders_by_price.items()):
                if oid in ids:
                    del self.placed_orders_by_price[px]

    # ---------------- Helpers ----------------

    def _own_volume_at(self, price, side):
        return sum(o.volume for o in self.all_active_orders if o.price == price and o.side == side)

    def _target_volume_for_price(self, price_2dec: float) -> int:
        """Детерминированный 100k–1M на уровень, чтобы цель не «плавала» между тиками."""
        seed = (hash((self.agent_id, round(price_2dec, 2))) & 0xffffffff)
        r = random.Random(seed)
        return r.randint(100_000, 1_000_000)

    def _ttl_by_distance_i(self, i: int) -> float:
        """
        TTL зависит от расстояния от mid:
        ближе — короче, глубже — дольше. Диапазоны подобраны против «тотальной дырки».
        """
        if i < 15:
            return random.uniform(6.0, 12.0)
        elif i < 60:
            return random.uniform(12.0, 25.0)
        elif i < 120:
            return random.uniform(20.0, 40.0)
        else:
            return random.uniform(30.0, 60.0)

    def _place(self, side, price, volume, now):
        o = Order(
            order_id=uuid.uuid4().hex,
            agent_id=self.agent_id,
            side=side,
            price=price,
            volume=volume,
            order_type=OrderType.LIMIT
        )
        o.timestamp = now
        o.ttl = None  # по умолчанию бессрочно; для глубины выставим ниже
        self.all_active_orders.append(o)
        return o

    # ---------------- Core ----------------

    def generate_orders(self, order_book, market_context=None):
        now = time.time()

        # 1) истекшие по индивидуальному ttl
        self.cancel_expired_orders(order_book)

        # 2) бутстрап из БД
        if self.bootstrap_active:
            out = []
            batch_limit = 2000
            while self.bootstrap_queue and len(out) < batch_limit:
                side, px, vol = self.bootstrap_queue.pop(0)
                o = self._place(OrderSide.BID if side == 'bid' else OrderSide.ASK, float(px), float(vol), now)
                o.ttl = None  # бутстрап держим до естественных событий
                out.append(o)
            if not self.bootstrap_queue or now >= self.bootstrap_deadline:
                self.bootstrap_active = False
            return out

        # 3) цены
        bid = order_book._best_bid_price()
        ask = order_book._best_ask_price()
        if bid is None or ask is None:
            return []

        out = []
        mid = round((bid + ask) / 2.0, 5)

        # 4) чистим mid-зону
        self._cancel_mid_zone(order_book, mid, self.mid_radius)

        # 5) удержание узкого спреда как было
        spread = random.uniform(0.01, 0.02)
        bid_price = round(mid - spread / 2, 2)
        ask_price = round(mid + spread / 2, 2)
        strong_volume = max(10_000, min(self.capital * 0.00005, 25_000))

        if bid_price < mid and bid_price not in self.placed_orders_by_price:
            o = self._place(OrderSide.BID, bid_price, strong_volume, now)
            o.ttl = random.uniform(4.0, 10.0)
            self.placed_orders_by_price[bid_price] = o.order_id
            out.append(o)

        if ask_price > mid and ask_price not in self.placed_orders_by_price:
            o = self._place(OrderSide.ASK, ask_price, strong_volume, now)
            o.ttl = random.uniform(4.0, 10.0)
            self.placed_orders_by_price[ask_price] = o.order_id
            out.append(o)

        # 6) гарантированная глубина: от ±0.06 до ±2.00
        for i in range(self.start_i, self.levels + 1):
            offset = round(i * 0.01, 5)
            p_bid = round(mid - offset, 2)
            p_ask = round(mid + offset, 2)

            # --- BID ---
            target = self._target_volume_for_price(p_bid)
            have = self._own_volume_at(p_bid, OrderSide.BID)
            need = target - have
            if need > 0:
                place = min(self.place_chunk, need)
                o = self._place(OrderSide.BID, p_bid, place, now)
                o.ttl = self._ttl_by_distance_i(i)
                out.append(o)

            # --- ASK ---
            target = self._target_volume_for_price(p_ask)
            have = self._own_volume_at(p_ask, OrderSide.ASK)
            need = target - have
            if need > 0:
                place = min(self.place_chunk, need)
                o = self._place(OrderSide.ASK, p_ask, place, now)
                o.ttl = self._ttl_by_distance_i(i)
                out.append(o)

        return out

    # ---------------- Fills ----------------

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        self.position += fill_size if side == OrderSide.BID else -fill_size
        # если это был "спредовый" ордер, освободить слот
        px = round(float(fill_price), 2)
        if px in self.placed_orders_by_price and self.placed_orders_by_price[px] == order_id:
            del self.placed_orders_by_price[px]
        # удалить из активных
        self.all_active_orders = [o for o in self.all_active_orders if o.order_id != order_id]
