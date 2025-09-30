import os
import time
import uuid
import logging, builtins
import numpy as np
import random
import threading
import collections

import signal
import sys

import memory_logger

from collections import deque

from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from flask import Flask, send_from_directory, request, jsonify


# ── тута свечки ──────────────────────────────────────────
from candle_db import init_db, insert_candle, load_candles, fill_incomplete_candle
from models import Candle
# ── тута ликвидность ─────────────────────────────────────
from liquidity_db import init_db as init_liquidity_db, load_liquidity_state, save_liquidity_state
# ───тута кластера ────────────────────────────────
from trade_db import init_db as init_trdb, insert_batch as tr_insert_batch, load_trades as tr_load_trades
# ─────────────────────────────────────────────────────────

conn = init_db()
liquidity_conn = init_liquidity_db()

TRDB = init_trdb()
TR_BUF = collections.deque(maxlen=500_000)

USER_TRADES = deque(maxlen=10000)

# ── silence per-trade spam ─────────────
#orig_print = builtins.print
#builtins.print = lambda *a, **k: None               
#logging.disable(logging.CRITICAL)                  
# ─────────────────────────────────

# ─────────────────────────────────
from trade_logger import _flush as flush_trades
# ─────────────────────────────────


from order import Order, OrderSide, OrderType
from order_book import OrderBook
from market_context import MarketContextAdvanced
from market_maker import AdvancedMarketMaker
from retail_trader import RetailTrader
from trend_follower import TrendFollowerAgent
from strategist_agent import InstitutionalExecutor
#from liquidity_hunter import LiquidityHunterAgent
#from high_freq import HighFreqAgent
from impulsive_aggressor import ImpulsiveAggressorAgent
#from fair_value_anchor import FairValueAnchorAgent
#from spread_arbitrage import SpreadArbitrageAgent
from passive_depth_provider import PassiveDepthProvider
#from base_agent import BaseAgent
#from market_order_agent import MarketOrderAgent
#from liquidity_manager import LiquidityManagerAgent
#from liquidity_manipulator import LiquidityManipulator
from smart_money import SmartMoneyManager
#from momentum_igniter import SmartSidewaysSqueezeAgent
#from bank_agent import BankAgentManager
#from corporate_vwap_agent import CorporateVWAPAgent
from trend_impuls import ReversalAgent
from bank_agent_v2 import BankAgentManagerv2
# disbalance_agent import PressureDisbalanceUnit
from breakout_agent import BreakoutTrader
#from absorption_agent import AbsorptionReversalAgent
#from trend_agent import TrendAgent

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger("OrderBookServer")

# --- Flask + SocketIO ---
app = Flask(__name__, static_folder='.', template_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*")

# ---- trades -> SQLite flush task ----
def trade_flush_task():
    while True:
        socketio.sleep(0.05)
        if not TR_BUF:
            continue
        batch = []
        while TR_BUF and len(batch) < 5000:
            batch.append(TR_BUF.popleft())
        # trade_db.insert_batch
        tr_insert_batch(TRDB, batch)

# --- Книга и стартовые лимитные ордера ---
order_book = OrderBook()
order_book.market_context = MarketContextAdvanced()

# --- Начальная ликвидность: 6 уровней по 20 ---
def inject_initial_liquidity():
    base = 100.0
    for i in range(1, 7):
        order_book.add_order(Order(
            order_id=str(uuid.uuid4()), agent_id=f"seed_buy_{i}",
            side=OrderSide.BID,
            volume=20.0,
            price=round(base - i * 0.5, 2),
            order_type=OrderType.LIMIT, ttl=None
        ))
        order_book.add_order(Order(
            order_id=str(uuid.uuid4()), agent_id=f"seed_sell_{i}",
            side=OrderSide.ASK,
            volume=20.0,
            price=round(base + i * 0.5, 2),
            order_type=OrderType.LIMIT, ttl=None
        ))
    seed_order_buy = Order(
        order_id=str(uuid.uuid4()),
        agent_id="seed_buy",
        side=OrderSide.BID,
        volume=10.0,
        price=99.50,
        order_type=OrderType.LIMIT,
        ttl=None
    )
    seed_order_sell = Order(
        order_id=str(uuid.uuid4()),
        agent_id="seed_sell",
        side=OrderSide.ASK,
        volume=10.0,
        price=100.50,
        order_type=OrderType.LIMIT,
        ttl=None
    )
    order_book.add_order(seed_order_buy)
    order_book.add_order(seed_order_sell)

def graceful_shutdown(*args):
    logger.info("Saving order book state before shutdown...")
    snapshot = order_book.get_order_book_snapshot(depth=1000)
    save_liquidity_state(liquidity_conn, {
        'bids': {entry['price']: entry['volume'] for entry in snapshot['bids']},
        'asks': {entry['price']: entry['volume'] for entry in snapshot['asks']},
    })
    logger.info("Shutdown complete.")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


# Загружаем ликвидность
saved_liquidity = load_liquidity_state(liquidity_conn)


market_phases_history = []
_last_phase_name = None
_last_phase_start = None
_last_microphase = None


# --- Инициализация начальных сделок ---
def inject_initial_trades():
    logger.info("[Init] Injecting initial market orders to trigger first trades")
    market_orders = [
        Order(str(uuid.uuid4()), "seeder_m1", OrderSide.BID, 2.0, None, OrderType.MARKET, None),
        Order(str(uuid.uuid4()), "seeder_m2", OrderSide.ASK, 1.0, None, OrderType.MARKET, None),
        Order(str(uuid.uuid4()), "seeder_m3", OrderSide.BID, 1.0, None, OrderType.MARKET, None),
        Order(str(uuid.uuid4()), "seeder_m4", OrderSide.ASK, 2.0, None, OrderType.MARKET, None),
    ]
    for mo in market_orders:
        order_book.add_order(mo)

# --- Список агентов ---

retail = RetailTrader(agent_id="retail1", capital=20_000_000.0, num_subagents=50)
trend  = TrendFollowerAgent(agent_id="trend1", capital=1_000_000.0, num_subagents=100)
#hf1    = LiquidityHunterAgent(agent_id="hf1", capital=4_000_000.0)
#hf2    = HighFreqAgent(agent_id="hf2", capital=2_000_000.0)
aggressor = ImpulsiveAggressorAgent(agent_id="imp1", capital=5_500_000.0, num_subagents=100)
#anchor = FairValueAnchorAgent(agent_id="anchor1", capital=50_000_000.0)
#arb = SpreadArbitrageAgent(agent_id="arb1", capital=1_000_000.0)
depth_provider = PassiveDepthProvider(agent_id="depth1", capital=1.0)
#base_agent = BaseAgent(agent_id="base1", total_capital=6_000_000.0, participants_per_agent=1_000)
#market_order_agent = MarketOrderAgent(agent_id="market_order1", capital=10_500_000.0)
#liquidity_manipulator = LiquidityManipulator(agent_id="manipulator1", capital=10_000_000.0)
smart_money = SmartMoneyManager(agent_id="smart", total_capital=1_000_000_000.0)
#momentum_igniter = SmartSidewaysSqueezeAgent(agent_id="momentum1", capital=20_000_000.0)
#bank_agent = BankAgentManager(agent_id="bank1", total_capital=1_000_000_000.0)
market_maker = AdvancedMarketMaker(agent_id="mm1", capital=70_000_000.0)
#corp_vwap = CorporateVWAPAgent(agent_id="corp_vwap", total_capital=1_000_000.0, legs=5)
#liquidity_manager = LiquidityManagerAgent(agent_id="liquidity_manager1", capital=1_000_000.0)
trend_imp = ReversalAgent(agent_id="trend_imp1", capital=1_000_000.0, num_subagents=100)
bank_agent_v2 = BankAgentManagerv2(agent_id="bankadv1", total_capital=1_000_000_000.0)
#disbalance_agent = PressureDisbalanceUnit(agent_id="disbalance", capital=50_000_000.0)
breakout_agent = BreakoutTrader(agent_id="breakout1", capital=40_000_000.0)
#absorption_agent = AbsorptionReversalAgent(agent_id="absorber1", capital=40_000_000.0)
#trend_agent = TrendAgent(agent_id="ICT1", capital=10_000_000.0)
exec_mgr = InstitutionalExecutor(agent_id="exec1", capital=7_000_000.0, num_subagents=30)



AGENTS = [
    market_maker,
    retail,
    trend,
    #hf1,
    #hf2,
    aggressor,
    #anchor,
    #  arb,
    depth_provider,
    #base_agent,
    #market_order_agent,
    #liquidity_manager,
    #liquidity_manipulator,
    smart_money,
    # momentum_igniter,
    #   bank_agent,
    #corp_vwap,
    trend_imp,
    bank_agent_v2,
    #disbalance_agent,
    breakout_agent,
    # absorption_agent,
    #trend_agent,
    exec_mgr
]

# Если ликвидности нет — значит первый запуск, инжектим
BOOTSTRAP_UNTIL = None

if saved_liquidity['bids'] or saved_liquidity['asks']:
    # НЕ добавляем ордера напрямую в order_book.
    # Просим провайдера глубины переиграть их под своим agent_id.
    depth_provider.begin_bootstrap(saved_liquidity, seconds=15.0)
    BOOTSTRAP_UNTIL = time.time() + 15.0
else:
    inject_initial_liquidity()

def warmup_agents(conn, agents, n=10):
    now = int(time.time())
    candles = load_candles(conn, '1m', now - n*60, now)
    if not candles:
        return
    mids = [(c.high + c.low) / 2 for c in candles]

    for mid in mids:
        for agent in agents:
            # простые price_history
            if hasattr(agent, "price_history"):
                agent.price_history.append(mid)
            # EMA у банковского
            if hasattr(agent, "update_emas"):
                agent.update_emas(mid)
            if hasattr(agent, "prev_mid_prices"):
                agent.prev_mid_prices.append(mid)
            # волатильность/ATR
            if hasattr(agent, "recent_volatility"):
                agent.recent_volatility.append(abs(mid - mids[-1]))  # примитивный шаг
    # для маркетмейкера
    for agent in agents:
        if hasattr(agent, "update_market_structure"):
            try:
                agent.update_market_structure(conn)
            except Exception as e:
                print(f"MarketMaker warmup error: {e}")

# Вызов сразу после AGENTS = [...]
warmup_agents(conn, AGENTS, n=10)


# ==== USER ACCOUNT ====
USER_ID = "terminal-ui"

class Account:
    MMR = 0.005  # maintenance margin rate 0.5%

    def __init__(self, balance=10_000.0, leverage=20, mode="cross"):
        self.balance = float(balance)
        self.leverage = int(leverage)
        self.mode = mode  # "cross" | "isolated"
        self.position_qty = 0.0      # +long, -short (в контрактах)
        self.entry_price = None
        self.realized_pnl = 0.0
        self.last_fee_rate = 0.0
        self.fee_paid = 0.
        self.fee_schedule = {"maker": 0.0002, "taker": 0.0004}

    # mark = mid(bid,ask) если есть, иначе last_trade
    def mark_price(self):
        bid = order_book._best_bid_price()
        ask = order_book._best_ask_price()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return order_book.last_trade_price or 100.0

    def notional(self, px=None):
        px = px if px is not None else self.mark_price()
        return abs(self.position_qty) * float(px)

    def upnl(self, px=None):
        if self.entry_price is None or self.position_qty == 0:
            return 0.0
        px = px if px is not None else self.mark_price()
        if self.position_qty > 0:   # long
            return (px - self.entry_price) * self.position_qty
        else:                       # short
            return (self.entry_price - px) * abs(self.position_qty)

    def equity(self, px=None):
        return self.balance + self.realized_pnl + self.upnl(px)

    def used_margin(self, px=None):
        # используем IM = N/L только по открытой позиции
        return self.notional(px) / max(self.leverage, 1)

    def available_cross(self, px=None):
        px = px if px is not None else self.mark_price()
        if self.mode == "isolated":
            free_bal = self.balance + self.realized_pnl
            return free_bal - self.used_margin(px)
        return self.equity(px) - self.used_margin(px)

    def liquidation_price(self):
        if self.position_qty == 0 or self.entry_price is None:
            return None
        q = abs(self.position_qty)
        px = self.mark_price()
        im = self.notional(self.entry_price) / max(self.leverage, 1)
        mmr = self.MMR

        # изолированная: margin = IM
        # кросс: margin = IM + (equity - used_margin)  (добавляется свободка)
        if self.mode == "isolated":
            margin = im
        else:
            margin = im + max(self.available_cross(px), 0.0)

        if self.position_qty > 0:  # long
            # margin/qty + Pliq - entry = Pliq*mmr  =>  Pliq = (entry - margin/qty)/(1-mmr)
            return (self.entry_price - margin / q) / max(1.0 - mmr, 1e-9)
        else:                       # short
            # margin/qty + entry - Pliq = Pliq*mmr  =>  Pliq = (entry + margin/qty)/(1+mmr)
            return (self.entry_price + margin / q) / (1.0 + mmr)

    def can_place(self, side, qty, px, order_type):
        qty = float(qty);
        px = float(px)

        ref_px = px if order_type == "limit" else self.mark_price()
        required_im = (qty * ref_px) / max(self.leverage, 1)
        max_notional = self.leverage * max(self.balance + self.realized_pnl, 0.0)

        if qty * ref_px > max_notional:
            return False, f"Превышен максимум: {qty * ref_px:.2f} > {max_notional:.2f}"

        if qty <= 0:
            return False, "Неверный объём"
        ref_px = px if order_type == "limit" else self.mark_price()
        required_im = (qty * ref_px) / max(self.leverage, 1)

        if self.mode == "isolated":
            free_bal = self.balance + self.realized_pnl
            if required_im > free_bal:
                return False, f"Недостаточно баланса: нужно {required_im:.2f}, есть {free_bal:.2f}"
        else:
            if required_im > self.equity(ref_px):
                return False, f"Недостаточно equity: нужно {required_im:.2f}, есть {self.equity(ref_px):.2f}"
        return True, ""

    def apply_fill(self, side, price, qty, is_taker=True):
        price = float(price)
        qty = float(qty)
        signed = qty if side == OrderSide.BID else -qty

        # комиссия по роли
        fee_rate = self.fee_schedule["taker"] if is_taker else self.fee_schedule["maker"]
        fee = price * qty * fee_rate
        self.last_fee_rate = fee_rate
        self.balance -= fee
        self.fee_paid += fee

        # добавление/закрытие позиции
        if self.position_qty == 0 or (self.position_qty > 0 and signed > 0) or (self.position_qty < 0 and signed < 0):
            new_qty = self.position_qty + signed
            if self.entry_price is None:
                self.entry_price = price
            else:
                wgt = abs(self.position_qty)
                self.entry_price = (self.entry_price * wgt + price * abs(signed)) / (wgt + abs(signed))
            self.position_qty = new_qty
        else:
            close_qty = min(abs(self.position_qty), abs(signed))
            pnl = (price - self.entry_price) * close_qty if self.position_qty > 0 else (
                                                                                                   self.entry_price - price) * close_qty
            self.realized_pnl += pnl
            self.position_qty += signed
            if self.position_qty == 0:
                self.entry_price = None

    def snapshot(self):
        px = self.mark_price()
        return {
            "mode": self.mode,
            "leverage": self.leverage,
            "balance": round(self.balance, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "upnl": round(self.upnl(px), 2),
            "equity": round(self.equity(px), 2),
            "used_margin": round(self.used_margin(px), 2),
            "available": round(self.available_cross(px), 2),
            "position_qty": self.position_qty,
            "entry_price": self.entry_price,
            "mark_price": round(px, 2),
            "liq_price": self.liquidation_price(),
            "max_position_notional": round(max(self.leverage,1) * max(self.balance + self.realized_pnl,0.0), 2),
            "fee_rate": self.last_fee_rate,
            "fee_paid": round(self.fee_paid, 2),
            "last_fee_rate": self.last_fee_rate,
            "margin_mode": self.mode,
            "liq_warning": (self.liquidation_price(), self.mark_price()),
            "notional": round(self.notional(px), 2),
        }

account = Account()

# ==== TP/SL conditional triggers (OCO) ====
_conditional = []


def _place_conditional(owner, side_close, tp=None, sl=None, trailing=None, qty=0.0, trigger_by="mark"):
    """
    Создаёт TP/SL/Trailing условные ордера.
    side_close: "sell" для закрытия long, "buy" для закрытия short
    tp/sl — абсолютные цены, trailing — offset в тех же единицах
    qty — объём для закрытия
    """
    import uuid
    oco_id = uuid.uuid4().hex

    if tp is not None:
        _conditional.append({
            "owner": owner,
            "side": side_close,
            "type": "tp",
            "trigger": float(tp),
            "trail_offset": None,
            "qty": float(qty),
            "oco": oco_id,
            "reduce_only": True,
            "trigger_by": trigger_by
        })
    if sl is not None:
        _conditional.append({
            "owner": owner,
            "side": side_close,
            "type": "sl",
            "trigger": float(sl),
            "trail_offset": None,
            "qty": float(qty),
            "oco": oco_id,
            "reduce_only": True,
            "trigger_by": trigger_by
        })
    if trailing is not None:
        # trailing стоп: стартовый trigger зависит от текущей цены
        mark = account.mark_price()
        if side_close == "sell":   # закрытие long
            trigger = mark - trailing
        else:                      # закрытие short
            trigger = mark + trailing
        _conditional.append({
            "owner": owner,
            "side": side_close,
            "type": "trailing",
            "trigger": trigger,
            "trail_offset": float(trailing),
            "qty": float(qty),
            "oco": oco_id,
            "reduce_only": True,
            "trigger_by": trigger_by
        })

    socketio.emit("conditional_update", _conditional)


def _eval_and_fire_conditionals():
    """
    Проверяет условия TP/SL/Trailing и исполняет при срабатывании.
    """
    if not _conditional:
        return

    last = order_book.last_trade_price or account.mark_price()
    mark = account.mark_price()
    fired_indices = []

    for i, c in enumerate(list(_conditional)):  # копия для безопасного удаления
        ref = mark if c["trigger_by"] == "mark" else last

        # trailing stop: подтягиваем trigger
        if c["type"] == "trailing" and c["trail_offset"] is not None:
            if c["side"] == "sell":  # закрытие long
                new_trigger = max(c["trigger"], mark - c["trail_offset"])
            else:                    # закрытие short
                new_trigger = min(c["trigger"], mark + c["trail_offset"])
            c["trigger"] = new_trigger

        # reduce-only: проверяем что есть позиция
        if account.position_qty == 0:
            continue

        # Логика направления
        cond = False
        if c["side"] == "sell":      # закрытие long
            if c["type"] in ("tp", "trailing"):
                cond = ref >= c["trigger"]
            elif c["type"] == "sl":
                cond = ref <= c["trigger"]
        else:                         # закрытие short (side=buy)
            if c["type"] in ("tp", "trailing"):
                cond = ref <= c["trigger"]
            elif c["type"] == "sl":
                cond = ref >= c["trigger"]

        if cond:
            qty = min(abs(account.position_qty), c["qty"])
            if qty <= 0:
                continue

            side = OrderSide.ASK if c["side"] == "sell" else OrderSide.BID
            px = order_book._best_bid_price() if side == OrderSide.ASK else order_book._best_ask_price()
            if px is None:
                logger.warning(f"[TP/SL] Trigger fired but no liquidity, skipping (side={side}, qty={qty})")
                continue

            # Прямое закрытие позиции
            account.apply_fill(side, px, qty, is_taker=True)


            # Событие трейда для фронта
            trade = {
                "price": px,
                "volume": qty,
                "side": "sell" if side == OrderSide.ASK else "buy",
                "initiator_side": "buy" if (side == OrderSide.BID) else "sell",
                "taker": USER_ID,
                "maker": "system"
            }
            socketio.emit("trade", trade)
            socketio.emit("account_update", account.snapshot())
            socketio.emit("positions_update", account.snapshot())

            logger.info(
                f"[TP/SL] Trigger fired: type={c['type']}, side={c['side']}, trigger={c['trigger']}, ref={ref}, qty={qty}, px={px}"
            )
            fired_indices.append(i)

            # Если OCO — убираем все ордера этой группы
            oco_id = c["oco"]
            _conditional[:] = [x for x in _conditional if x["oco"] != oco_id]

    # убрать сработавшие (OCO уже удалил группу)
    for idx in sorted(fired_indices, reverse=True):
        if idx < len(_conditional):
            _conditional.pop(idx)

    socketio.emit("conditional_update", _conditional)
    socketio.emit("positions_update", account.snapshot())



def push_positions():
    socketio.emit("positions_update", account.snapshot())
    socketio.start_background_task(schedule_push)

def schedule_push():
    while True:
        socketio.sleep(3)
        socketio.emit("positions_update", account.snapshot())
socketio.start_background_task(schedule_push)

# --- OHLC (свечи) ---

class PhaseTracker:
    def __init__(self, maxlen=300):
        self.current_phase = None
        self.start_time = None
        self.tracked_phases = deque(maxlen=maxlen)

    def update(self, market_context):
        now = int(time.time())
        phase = market_context.phase.name if market_context and market_context.phase else "undefined"
        micro = market_context.phase.reason.get("microphase", "") if market_context and market_context.phase and market_context.phase.reason else ""

        if self.current_phase != (phase, micro):
            if self.current_phase:
                self.tracked_phases.append({
                    "start": self.start_time,
                    "end": now,
                    "phase": self.current_phase[0],
                    "microphase": self.current_phase[1]
                })
            self.current_phase = (phase, micro)
            self.start_time = now

    def export(self):
        now = int(time.time())
        exported = list(self.tracked_phases)
        if self.current_phase:
            exported.append({
                "start": self.start_time,
                "end": now,
                "phase": self.current_phase[0],
                "microphase": self.current_phase[1]
            })
        return exported

phase_tracker = PhaseTracker()

USER_ORDERS = {}  # order_id -> {"is_taker": bool}



# --- HTTP ---
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/styles.css')
def css():
    return send_from_directory(os.getcwd(), 'styles.css')

@app.route('/scripts.js')
def js():
    return send_from_directory(os.getcwd(), 'scripts.js')

@app.route('/chart/<path:filename>')
def serve_chart(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'chart'), filename)

@app.route('/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'sounds'), filename)

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'images'), filename)

@app.route("/profile")
def profile_page():
    return send_from_directory(".", "profile.html")

@socketio.on('get_user_trades')
def handle_get_user_trades():
    emit("user_trades", list(USER_TRADES)[-100:])

@socketio.on('get_account')
def handle_get_account():
    emit("account_update", account.snapshot())

# --- Socket.IO ---
@socketio.on('connect')
def on_connect():
    sid = request.sid
    logger.info(f"Client connected: {sid}")
    emit("orderbook_update", order_book.get_order_book_snapshot(depth=13))
    hist = []
    for t in list(order_book.trade_history)[-200:]:
        side_str = str(t.get("taker_side", "")).lower()
        t = dict(t)
        t["initiator_side"] = "buy" if side_str in ("buy", "bid") else "sell"
        hist.append(t)
    emit("history", hist)
    emit("account_update", account.snapshot())
    emit("conditional_update", _conditional)

@socketio.on('set_conditionals')
def handle_set_conditionals(data):
    try:
        tp = data.get('tp')
        sl = data.get('sl')
        trig_by = data.get('trigger_by', 'mark')

        # нет позиции — просто отдать текущее
        if abs(account.position_qty) <= 0:
            emit("conditional_update", _conditional)
            return

        # противоположная сторона для закрытия
        side_close = 'sell' if account.position_qty > 0 else 'buy'
        qty = abs(account.position_qty)

        # убрать старые reduce-only условки этого пользователя
        owner = USER_ID
        oco_ids = {c["oco"] for c in _conditional if c.get("owner")==owner and c.get("reduce_only")}
        del oco_ids  # инфо-поле, чистим всю группу ниже
        _conditional[:] = [
            c for c in _conditional
            if not (c.get("owner")==owner and c.get("reduce_only"))
        ]

        # поставить новые, если заданы
        if tp is not None or sl is not None:
            _place_conditional(
                owner, side_close,
                tp=float(tp) if tp is not None else None,
                sl=float(sl) if sl is not None else None,
                qty=qty,
                trigger_by=trig_by
            )
        # разослать актуальное состояние
        socketio.emit("conditional_update", _conditional)
    except Exception as e:
        emit("error", {"message": f"set_conditionals: {e}"})


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    logger.info(f"Client disconnected: {sid}")

@socketio.on('add_order')
def handle_add_order(data):
    try:
        side = OrderSide.BID if data.get('side') == 'buy' else OrderSide.ASK
        otype = data.get('order_type', 'limit')
        order_type = OrderType.LIMIT if otype == 'limit' else OrderType.MARKET

        qty = float(data.get('volume'))
        px = float(data.get('price')) if data.get('price') is not None else account.mark_price()

        ok, reason = account.can_place(side, qty, px, otype)
        if not ok:
            emit("error", {"message": f"Order rejected: {reason}"})
            return

        # ордер
        order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=data.get('agent_id') or USER_ID,
            side=side,
            volume=qty,
            price=None if order_type == OrderType.MARKET else float(data.get('price')),
            order_type=order_type,
            ttl=None
        )
        trades = order_book.add_order(order)

        if trades:
            for t in trades:
                notify_agent_fill(t)
                # ---- буфер для бдшки ----
                side = 'buy' if str(t.get('taker_side', '')).lower() in ('buy', 'bid') else 'sell'
                TR_BUF.append((int(time.time()), float(t['price']), float(t['volume']), side))
                side_str = str(t.get("taker_side", "")).lower()
                t["initiator_side"] = "buy" if side_str in ("buy", "bid") else "sell"
                # ---------------------------
                socketio.emit("trade", t)
                for tf, cm in candle_managers.items():
                    cm.update(t['price'], t['volume'])
                    if cm.current_candle:
                        insert_candle(conn, tf, cm.current_candle)
                        socketio.emit("candles", [cm.current_candle], broadcast=True)

        is_taker = (order_type == OrderType.MARKET)
        if (data.get('agent_id') or USER_ID) == USER_ID:
            USER_ORDERS[order.order_id] = {"is_taker": is_taker}

        # TP/SL как условные (reduce-only)
        if data.get('tpsl_enabled'):
            trig_by = data.get('trigger_by', 'mark')
            tp = float(data['tp']) if data.get('tp') not in (None, "", "null") else None
            sl = float(data['sl']) if data.get('sl') not in (None, "", "null") else None

            # направлением закрытия будет противоположная сторона
            side_close = 'sell' if side == OrderSide.BID else 'buy'

            # берём реальный текущий объём позиции (если он есть), иначе исходный объём ордера
            cond_qty = abs(account.position_qty) if account.position_qty != 0 else qty

            _place_conditional(
                USER_ID,
                side_close,
                tp=tp,
                sl=sl,
                qty=cond_qty,
                trigger_by=trig_by
            )
            logger.info(
                f"[TP/SL] Conditional placed: side_close={side_close}, tp={tp}, sl={sl}, qty={cond_qty}, trig_by={trig_by}")
        emit("confirmation", {"message": f"Order {order.order_id} accepted"})
        emit("account_update", account.snapshot())
        socketio.emit("positions_update", account.snapshot())
    except Exception as e:
        logger.error(f"add_order error: {e}")
        emit("error", {"message": str(e)})

@socketio.on('set_account')
def handle_set_account(data):
    try:
        lv = int(str(data.get('leverage', account.leverage)).replace('x',''))
        mode = data.get('mode', account.mode)
        account.leverage = max(1, min(125, lv))
        account.mode = "isolated" if str(mode).lower().startswith("изол") or mode=="isolated" else "cross"
        emit("account_update", account.snapshot())
    except Exception as e:
        emit("error", {"message": f"set_account: {e}"})

@socketio.on("set_conditionals")
def on_set_conditionals(data):
    tp = data.get("tp")
    sl = data.get("sl")
    trigger_by = data.get("trigger_by", "mark")

    # закрывающая сторона зависит от текущей позиции
    side_close = None
    if account.position_qty > 0:
        side_close = "sell"
    elif account.position_qty < 0:
        side_close = "buy"

    if side_close is None or account.position_qty == 0:
        return  # нет позиции → нечего ставить TP/SL

    qty = abs(account.position_qty)

    _place_conditional(
        owner=USER_ID,
        side_close=side_close,
        tp=tp,
        sl=sl,
        qty=qty,
        trigger_by=trigger_by
    )


@socketio.on('cancel_order')
def handle_cancel_order(data):
    oid = data.get('order_id')
    if not oid:
        emit("error", {"message": "Missing order_id"})
        return
    if order_book.cancel_order(oid):
        emit("confirmation", {"message": f"Order {oid} cancelled"})
        emit("account_update", account.snapshot())
    else:
        emit("error", {"message": f"Order {oid} not found or inactive"})

@socketio.on('close_position')
def handle_close_position(data):
    if account.position_qty == 0:
        emit("error", {"message": "Нет открытой позиции"})
        return

    px = order_book._best_bid_price() if account.position_qty > 0 else order_book._best_ask_price()
    if px is None:
        emit("error", {"message": "Нет ликвидности для закрытия"})
        return

    qty = abs(account.position_qty)
    pos_side = "buy" if account.position_qty > 0 else "sell"
    side = OrderSide.ASK if account.position_qty > 0 else OrderSide.BID

    # --- PnL ---
    if account.entry_price is not None:
        if account.position_qty > 0:  # long закрывался
            trade_pnl = (px - account.entry_price) * qty
        else:  # short закрывался
            trade_pnl = (account.entry_price - px) * qty
    else:
        trade_pnl = 0.0

    # Прямое схлопывание через account.apply_fill (reduce-only)
    account.apply_fill(side, px, qty, is_taker=True)

    # --- Запись итоговой сделки в историю ---
    USER_TRADES.append({
        "ts": int(time.time()),
        "side": "buy" if pos_side == "buy" else "sell",
        "price": px,
        "qty": qty,
        "fee_delta": account.last_fee_rate * px * qty * -1,
        "realized_delta": trade_pnl,
        "equity_after": account.equity()
    })
    socketio.emit("user_trades", list(USER_TRADES)[-100:])

    # Формируем простой трейд только для фронта
    trade = {
        "price": px,
        "volume": qty,
        "side": "sell" if side == OrderSide.ASK else "buy",
        "initiator_side": "buy" if (side == OrderSide.BID) else "sell",
        "taker": USER_ID,
        "maker": "system"
    }

    emit("confirmation", {"message": "Позиция закрыта"})
    socketio.emit("account_update", account.snapshot())
    socketio.emit("positions_update", account.snapshot())
    socketio.emit("trade", trade)

@socketio.on('close_partial')
def handle_close_partial(data):
    if account.position_qty == 0:
        emit("error", {"message": "Нет открытой позиции"})
        return

    try:
        qty = float(data.get("qty", 0))
    except:
        emit("error", {"message": "Неверный объём"})
        return

    if qty <= 0 or qty > abs(account.position_qty):
        emit("error", {"message": f"Неверный объём: {qty}"})
        return

    px = order_book._best_bid_price() if account.position_qty > 0 else order_book._best_ask_price()
    if px is None:
        emit("error", {"message": "Нет ликвидности для закрытия"})
        return

    pos_side = "buy" if account.position_qty > 0 else "sell"
    side = OrderSide.ASK if account.position_qty > 0 else OrderSide.BID
    account.apply_fill(side, px, qty, is_taker=True)


    # --- Запись итоговой частичной сделки в историю ---
    USER_TRADES.append({
        "ts": int(time.time()),
        "side": pos_side,
        "price": px,
        "qty": qty,
        "fee_delta": account.last_fee_rate * px * qty * -1,
        "realized_delta": account.realized_pnl,
        "equity_after": account.equity()
    })
    socketio.emit("user_trades", list(USER_TRADES)[-100:])

    trade = {
        "price": px,
        "volume": qty,
        "side": "sell" if side == OrderSide.ASK else "buy",
        "initiator_side": "buy" if (side == OrderSide.BID) else "sell",
        "taker": USER_ID,
        "maker": "system"
    }

    emit("confirmation", {"message": f"Закрыто {qty} контрактов"})
    socketio.emit("account_update", account.snapshot())
    socketio.emit("positions_update", account.snapshot())
    socketio.emit("trade", trade)


@app.route("/candles", methods=["GET"])
def get_candles():
    tf = int(request.args.get("interval", 5))
    candles = candle_managers[tf].get_candles()
    return jsonify([
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume
        } for c in candles
    ])

@app.route("/api/trades")
def api_trades():
    try:
        ts_from = int(request.args.get("from", str(int(time.time()) - 3600)))
        ts_to   = int(request.args.get("to",   str(int(time.time()))))
        limit   = int(request.args.get("limit", "2000"))
        side    = request.args.get("side")      # 'buy' | 'sell' | None
        order   = request.args.get("order", "asc")
    except Exception:
        return jsonify({"error": "bad params"}), 400
    data = tr_load_trades(TRDB, ts_from, ts_to, limit, side, order)
    return jsonify(data)


#@socketio.on('candles')
#def handle_candles(data=None):
    # test = [
    #      {"timestamp": 1, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 10},
    #       {"timestamp": 2, "open": 102, "high": 106, "low": 101, "close": 104, "volume": 15},
    #   ]
#   emit('candles', test)


@app.route("/market_phases")
def get_market_phases():
    # последние 200 фаз, чтобы не грузить лишнего
    return jsonify(phase_tracker.export())



# --- Fill уведомление ---
def notify_agent_fill(trade):
    price = trade['price']
    volume = trade['volume']
    for agent in AGENTS:
        if agent.agent_id == trade['buy_agent']:
            agent.on_order_filled(trade['buy_order_id'], price, volume, OrderSide.BID)
        if agent.agent_id == trade['sell_agent']:
            agent.on_order_filled(trade['sell_order_id'], price, volume, OrderSide.ASK)
        # учёт терминала с ролью maker/taker
        if trade['buy_agent'] == USER_ID:
            meta = USER_ORDERS.get(trade['buy_order_id'], {"is_taker": True})
            before_realized = account.realized_pnl
            before_fee = account.fee_paid
            account.apply_fill(OrderSide.BID, trade['price'], trade['volume'], is_taker=meta["is_taker"])
            realized_delta = account.realized_pnl - before_realized
            fee_delta = account.fee_paid - before_fee
            #_log_user_trade("buy", trade['price'], trade['volume'], meta["is_taker"],
                            #fee_delta, realized_delta, account.equity())
            _emit_account_stats(socketio)

        if trade['sell_agent'] == USER_ID:
            meta = USER_ORDERS.get(trade['sell_order_id'], {"is_taker": True})
            before_realized = account.realized_pnl
            before_fee = account.fee_paid
            account.apply_fill(OrderSide.ASK, trade['price'], trade['volume'], is_taker=meta["is_taker"])
            realized_delta = account.realized_pnl - before_realized
            fee_delta = account.fee_paid - before_fee
            #_log_user_trade("sell", trade['price'], trade['volume'], meta["is_taker"],
                            #fee_delta, realized_delta, account.equity())
            _emit_account_stats(socketio)

    socketio.emit("account_update", account.snapshot())
    socketio.emit("positions_update", account.snapshot())


# --- Matching и трансляция ---
def match_and_broadcast():
    global _last_phase_name, _last_phase_start, _last_microphase
    flush_trades.last = time.time()

    while True:
        try:
            order_book.tick()
            trades = order_book.match()

            # активируем условные TP/SL
            _eval_and_fire_conditionals()
            socketio.emit("account_update", account.snapshot())

            # ---------- агрегированная сводка раз в минуту ----------
            now = time.time()
            if now - getattr(flush_trades, "last", 0) >= 60:
                flush_trades(now)
                flush_trades.last = now

            # === ФАЗОВЫЙ КОНТЕКСТ ===
            if hasattr(order_book, "market_context"):
                ctx = order_book.market_context
                snapshot = order_book.get_order_book_snapshot()
                sweep_occurred = bool(trades)
                order_book.market_context.update(snapshot, sweep_occurred)
                phase_tracker.update(ctx)
                # Берём name и microphase
                phase = ctx.phase.name if ctx.phase else "undefined"
                micro = ctx.phase.reason.get("microphase", "") if ctx.phase and hasattr(ctx.phase, "reason") else ""
                now = int(time.time())
                if phase != _last_phase_name or micro != _last_microphase:
                    # Заканчиваем предыдущую фазу
                    if _last_phase_name is not None and _last_phase_start is not None:
                        market_phases_history.append({
                            "start": _last_phase_start,
                            "end": now,
                            "phase": _last_phase_name,
                            "microphase": _last_microphase or ""
                        })
                    _last_phase_name = phase
                    _last_phase_start = now
                    _last_microphase = micro

            if account.position_qty != 0:
                liq = account.liquidation_price()
                mark = account.mark_price()
                if (account.position_qty > 0 and mark <= liq) or (account.position_qty < 0 and mark >= liq):
                    # Ликвидация
                    order_book.add_order(Order(
                        order_id=str(uuid.uuid4()), agent_id=USER_ID,
                        side=OrderSide.ASK if account.position_qty > 0 else OrderSide.BID,
                        volume=abs(account.position_qty),
                        price=None, order_type=OrderType.MARKET, ttl=None
                    ))
                    socketio.emit("liquidation", {"liq_price": liq, "qty": abs(account.position_qty)})

            if trades:
                for t in trades:
                    notify_agent_fill(t)
                    side_str = str(t.get("taker_side", "")).lower()
                    t["initiator_side"] = "buy" if side_str in ("buy","bid") else "sell"
                    # ---- буфер для бдшки ----
                    side = 'buy' if t["initiator_side"] == 'buy' else 'sell'
                    TR_BUF.append((int(time.time()), float(t['price']), float(t['volume']), side))
                    # ---------------------------
                    socketio.emit("trade", t)  # Отправляем сделку клиенту
                    for tf, cm in candle_managers.items():
                        cm.update(t['price'], t['volume'])  # Обновляем свечку
                        if cm.current_candle:
                            insert_candle(conn, tf, cm.current_candle)  # Сохраняем последнюю свечку

            # Отправляем свечи
            for tf, cm in candle_managers.items():
                socketio.emit("candles", [
                    {
                        "timestamp": c.timestamp,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume
                    }
                    for c in cm.get_candles()
                ])

            socketio.emit("orderbook_update", order_book.get_order_book_snapshot(depth=13))

        except Exception as e:
            logger.error(f"match loop error: {e}")
            # пушим состояние аккаунта на фронт

        socketio.emit("account_update", account.snapshot())
        socketio.sleep(0.3)

def _log_user_trade(side, price, qty, is_taker, fee_delta, realized_delta, equity_after):
    USER_TRADES.append({
        "ts": time.time(),
        "side": side,
        "price": float(price),
        "qty": float(qty),
        "is_taker": is_taker,
        "fee_delta": float(fee_delta),
        "realized_delta": float(realized_delta),
        "equity_after": float(equity_after),
    })

def _calc_account_stats():
    trades = list(USER_TRADES)
    realized_total = sum(t["realized_delta"] for t in trades)
    fees_total = sum(t.get("fee_delta", 0.0) for t in trades)
    closed = [t for t in trades if t["realized_delta"] != 0]
    wins = sum(1 for t in closed if t["realized_delta"] > 0)
    losses = sum(1 for t in closed if t["realized_delta"] < 0)
    winrate = wins / max(1, (wins + losses))
    # max drawdown по equity
    peak = float("-inf")
    dd = 0.0
    for t in trades:
        eq = t["equity_after"]
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {
        "realized_total": round(realized_total, 2),
        "fees_total": round(fees_total, 2),
        "trades_count": len(trades),
        "closed_count": len(closed),
        "winrate": round(winrate, 4),
        "max_drawdown": round(dd, 2),
    }


def _emit_account_stats(socketio):
    socketio.emit("account_stats", _calc_account_stats())
    socketio.emit("user_trades", list(USER_TRADES)[-500:])

# --- Отправка свечей ---


# --- Агенты: разный ритм для каждого ---
def agents_loop():
    global BOOTSTRAP_UNTIL
    next_call = {a.agent_id: 0 for a in AGENTS}
    while True:
        now = time.time()
        for agent in AGENTS:
            # во время бутстрапа работает только провайдер глубины
            if 'BOOTSTRAP_UNTIL' in globals() and BOOTSTRAP_UNTIL and now < BOOTSTRAP_UNTIL:
                if not isinstance(agent, PassiveDepthProvider):
                    continue

            interval = 0.5
            if now >= next_call[agent.agent_id]:
                try:
                    if isinstance(agent, BreakoutTrader):
                        agent.update_market_structure(conn)

                    if 'AbsorptionReversalAgent' in globals() and isinstance(agent, AbsorptionReversalAgent):
                        candles = candle_managers[60].get_candles()  # 1m = 60s
                        agent.update_price_window(candles)

                    if getattr(agent, "supports_conn", False):
                        new_orders = agent.generate_orders(order_book, order_book.market_context, conn=conn)
                    else:
                        new_orders = agent.generate_orders(order_book, order_book.market_context)

                    for order in new_orders:
                        trades = order_book.add_order(order)
                        if trades:
                            for trade in trades:
                                notify_agent_fill(trade)

                                _emit_account_stats(socketio)
                                trade["initiator_side"] = "buy" if str(trade.get("taker_side", "")).lower() in ("buy", "bid") else "sell"
                                socketio.emit("trade", trade)  # отправить клиенту
                                for tf, cm in candle_managers.items():
                                    cm.update(trade['price'], trade['volume'])
                                    if cm.current_candle:
                                        insert_candle(conn, tf, cm.current_candle)
                except Exception as e:
                    logger.error(f"[Agent {agent.agent_id}] error: {e}")
                next_call[agent.agent_id] = now + interval

        if BOOTSTRAP_UNTIL:
            done = not getattr(depth_provider, "bootstrap_active", False)
            if done or time.time() >= BOOTSTRAP_UNTIL:
                BOOTSTRAP_UNTIL = None

        socketio.sleep(0.2)



class Candle:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

class CandleManager:
    def __init__(self, interval_seconds=5):
        self.interval = interval_seconds
        self.current_candle = None
        self.history = []

    def update(self, trade_price, trade_volume, trade_time=None):
        now = trade_time or time.time()
        bucket = int(now // self.interval) * self.interval

        if not self.current_candle or self.current_candle.timestamp != bucket:
            if self.current_candle:
                self.history.append(self.current_candle)
            self.current_candle = Candle(
                timestamp=bucket,
                open=self.current_candle.close if self.current_candle else trade_price,
                high=trade_price,
                low=trade_price,
                close=trade_price,
                volume=trade_volume
            )
        else:
            c = self.current_candle
            c.high = max(c.high, trade_price)
            c.low = min(c.low, trade_price)
            c.close = trade_price
            c.volume += trade_volume

    def get_candles(self, limit=1000):
        return self.history[-limit:] + ([self.current_candle] if self.current_candle else [])

    def tick(self, last_price=None):
        now = time.time()
        bucket = int(now // self.interval) * self.interval

        if not self.current_candle or self.current_candle.timestamp != bucket:
            if self.current_candle:
                self.history.append(self.current_candle)

            # Новый способ определения стартовой цены:
            ref_price = None
            if self.current_candle:
                ref_price = self.current_candle.close
            elif self.history:
                ref_price = self.history[-1].close
            elif last_price:
                ref_price = last_price
            else:
                ref_price = 100.0  # самый последний fallback

            self.current_candle = Candle(
                timestamp=bucket,
                open=ref_price,
                high=ref_price,
                low=ref_price,
                close=ref_price,
                volume=0.0
            )

def candle_tick_loop():
    while True:
        # Предполагаем, что best mid-price = средняя между bid/ask
        bid = order_book._best_bid_price()
        ask = order_book._best_ask_price()
        mid = (bid + ask) / 2 if bid and ask else None

        for cm in candle_managers.values():
            cm.tick(last_price=mid)

        socketio.sleep(1.0)



# --- Инициализация менеджеров таймфреймов ---
first_run = True

candle_managers = {
    tf: CandleManager(interval_seconds=tf) for tf in [1, 15, 60, 300, 3600]
}
now = int(time.time())
from_ts = 0
to_ts = now + 100000

if first_run:
    for tf, cm in candle_managers.items():
        cm.history = fill_incomplete_candle(cm.history, tf)  # Загружаем данные из базы
        print(f"[INFO] Loaded {len(cm.history)} candles for tf {tf}s")
    first_run = False  # После подгрузки, флаг
for tf, cm in candle_managers.items():
    cm.history = load_candles(conn, tf, from_ts, to_ts)
    print(f"[INFO] Loaded {len(cm.history)} candles for tf {tf}s")





def memory_logger_loop():
    tick_count = 0
    while True:
        tick_count += 1
        if tick_count % (60 * 2) == 0:
            memory_logger.log_memory_snapshot(label=f"Tick {tick_count}")
        socketio.sleep(0.5)  # даём управление другим задачам

# --- MAIN ---
if __name__ == '__main__':
    socketio.start_background_task(match_and_broadcast)
    socketio.start_background_task(agents_loop)
    socketio.start_background_task(candle_tick_loop)
    socketio.start_background_task(memory_logger_loop)
    socketio.start_background_task(trade_flush_task)
    logger.info("Starting server on http://0.0.0.0:8000 …")
    socketio.run(app, host='0.0.0.0', port=8000)