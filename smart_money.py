# smart_money.py
# Совместим с server.py: класс SmartMoneyManager(supports_conn, generate_orders, on_order_filled)

import time
import uuid
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Deque, Dict
from collections import deque

# проектные импорты
from order import Order, OrderSide, OrderType, quant
from candle_db import load_candles


# =============== утилиты ===============
def now_ts() -> int:
    return int(time.time())


def ou_step(x: float, theta: float, sigma: float) -> float:
    return (1 - theta) * x + sigma * random.gauss(0.0, 1.0)


def atr_from_candles(candles) -> float:
    if not candles:
        return 0.01
    rng = [max(c.high - c.low, 0.0) for c in candles]
    return max(sum(rng) / max(len(rng), 1), 0.01)


def microprice_from_snapshot(snap) -> Optional[float]:
    bids = snap.get("bids", [])
    asks = snap.get("asks", [])
    if not bids or not asks:
        return None
    pb, vb = bids[0]["price"], max(bids[0].get("volume", 0.0), 1e-9)
    pa, va = asks[0]["price"], max(asks[0].get("volume", 0.0), 1e-9)
    return (pa * vb + pb * va) / (vb + va)


def sum_depth(snap, side: OrderSide, levels: int = 10) -> float:
    book = snap.get("bids" if side == OrderSide.BID else "asks", [])
    return sum(float(e.get("volume", 0.0)) for e in book[:levels])


def top_of_book(snap) -> Tuple[float, float, float, float]:
    bids = snap.get("bids", [])
    asks = snap.get("asks", [])
    pb = bids[0]["price"] if bids else float("nan")
    vb = float(bids[0].get("volume", 0.0)) if bids else 0.0
    pa = asks[0]["price"] if asks else float("nan")
    va = float(asks[0].get("volume", 0.0)) if asks else 0.0
    return pb, vb, pa, va


# =============== дата-классы ===============
@dataclass
class ExecProfile:
    r_lo: float
    r_hi: float
    lob_beta: float
    ou_theta: float
    ou_sigma: float
    ping_prob: float
    ttl_min: int
    ttl_max: int
    accel: float


@dataclass
class SubTask:
    sub_id: str
    side: OrderSide
    notional: float
    started_at: int
    horizon_sec: int
    deadline_ts: Optional[int]
    arrival_price: float
    profile: ExecProfile
    scale: str  # 'micro'|'meso'|'macro'
    # динамика
    ou_x: float = 1.0
    last_action_ts: int = 0
    next_action_after: float = 0.0
    state: str = "idle"          # burst|idle
    window_end_ts: int = 0
    # полосы
    band_w: float = 0.0
    band_edge: Optional[float] = None
    # ASD
    phase: str = "accumulate"    # accumulate|sweep|distribute
    phase_until: int = 0
    accum_target: float = 0.0
    accum_done: float = 0.0
    sweep_budget: float = 0.0
    last_mid_on_sweep: float = 0.0
    post_sweep_check_ts: int = 0
    # волны
    waves_left: int = 1
    wave_base_price: float = 0.0
    sweep_block_until: int = 0
    # локальные гейты
    cool_off_until: int = 0
    market_disabled_until: int = 0
    # форма/факторы
    shape: str = "linear"                  # 'front'|'back'|'middle'|'linear'
    factor_load: Tuple[float, float] = (0.0, 0.0)
    rate_eps: float = 0.0
    # учёт
    executed: float = 0.0
    completed: bool = False
    base_mode_until: int = 0  # режим «базы»


@dataclass
class Regime:
    last_hour: int = -1
    macro_refractory_until: int = 0
    sweep_p: float = 0.18
    wave_count_mu: float = 2.0
    silent_lambda_per_hour: float = 12.0
    macro_cap_mult: float = 1.0


# =============== менеджер ===============
class SmartMoneyManager:
    """Исполнитель с параллельными потоками, волнами, полосами, анти-трейсом, откатами, ловушками и лёгким ММ."""
    supports_conn: bool = True

    # ---------- init ----------
    def __init__(self, agent_id: str, total_capital: float = 1_000_000_000.0, seed: Optional[int] = None):
        self.agent_id = agent_id
        self.total_capital = total_capital
        random.seed(seed if seed is not None else ((hash(agent_id) ^ now_ts()) & 0xFFFFFFFF))

        # память книги
        self.prev_pb = self.prev_vb = self.prev_pa = self.prev_va = None

        # сабтаски
        self.subs: List[SubTask] = []
        self.min_subs, self.max_subs = 3, 7

        # менеджер
        self.arrival_mid: Optional[float] = None
        self.cum_slippage = 0.0
        self.cum_executed = 0.0
        self.order_owner: Dict[str, str] = {}  # order_id -> sub_id

        # анти-трейс
        self.sign_hist: Deque[Tuple[int, int]] = deque(maxlen=1500)

        # история цены
        self.mid_hist: Deque[Tuple[int, float]] = deque(maxlen=4 * 3600)  # 4ч по секундам
        self.last_atr1h = 0.01
        self.last_atr5m = 0.01

        # окна
        self.macro_window_until = 0
        self.silent_until = 0

        # реверс-хедж
        self.reverse_quota = 0.0
        self.reverse_lock_until = 0
        self.reverse_earliest_ts = 0
        self._rq_t = now_ts()
        self._last_vol_1m = 0.0

        # ММ
        self.default_tick = 0.01
        self.mm_enabled = True
        self.inv = 0.0
        self.inv_max = 5_000_000.0
        self.mm_min_spread = 2 * self.default_tick
        self.mm_cooldown_until = 0

        # анти-линейность
        self.sign_roll: Deque[int] = deque(maxlen=900)  # ~15 мин
        self.run_len = 0
        self.prev_sign = 0
        self.flow_energy = 0.0

        # режим часа
        self.regime = Regime()

        # статистика часа
        self.hour_stats = {"sweeps": 0, "net_sign": 0, "max_run": 0}

        # латентные факторы
        self.F1 = 0.0
        self.F2 = 0.0

    # ---------- fills ----------
    def on_order_filled(self, order_id: str, price: float, volume: float, side: OrderSide):
        sub_id = self.order_owner.pop(order_id, None)

        # инвентарь
        self.inv += abs(volume) if side == OrderSide.BID else -abs(volume)

        # учёт сабтасков
        if sub_id and sub_id not in ("mm",):
            for s in self.subs:
                if s.sub_id == sub_id:
                    s.executed += abs(volume)
                    if s.phase == "accumulate":
                        s.accum_done += abs(volume)
                    s.notional = max(0.0, s.notional - abs(volume))
                    if s.notional <= 0.0:
                        s.completed = True
                    break

        # издержки
        if self.arrival_mid is not None:
            if side == OrderSide.BID:
                self.cum_slippage += max(0.0, price - self.arrival_mid) * volume
            else:
                self.cum_slippage += max(0.0, self.arrival_mid - price) * volume
        self.cum_executed += abs(volume)

    # ---------- main ----------
    def generate_orders(self, order_book, market_context, conn=None) -> List[Order]:
        now = now_ts()
        out: List[Order] = []

        # режим часа
        self._update_regime(now)

        # распад реверса
        dt = now - self._rq_t
        self._rq_t = now
        self.reverse_quota *= 0.5 ** (dt / 180.0)  # T1/2 ~ 3 мин

        # латентные факторы
        self.F1 = 0.95 * self.F1 + 0.15 * random.gauss(0, 1)
        self.F2 = 0.90 * self.F2 + 0.25 * random.gauss(0, 1)

        # снапшот
        snap = order_book.get_order_book_snapshot(depth=10)
        mid = microprice_from_snapshot(snap)
        if mid is None:
            return out
        if self.arrival_mid is None:
            self.arrival_mid = mid
        self._update_mid_hist(mid, now)
        self._update_atrs(conn, now)

        # тишина
        if now < self.silent_until:
            return out
        self._maybe_start_silent_window(now)

        # макро
        self._maybe_macro_window(now)
        macro_mode = now < self.macro_window_until

        # сабтаски
        self._maybe_activate_subs(order_book, snap, now)
        self.subs = [s for s in self.subs if not s.completed]

        # бюджет IS
        if not self._budgets_ok():
            return out

        # OFI и спред
        ofi, spread = self._compute_ofi(snap)

        # анти-трейс: охлаждаем только часть доминирующей стороны
        self._anti_trace_controller(now)

        # минутный объём
        vol_1m_global = self._estimate_1m_volume(order_book)
        self._last_vol_1m = vol_1m_global
        self.reverse_quota = min(self.reverse_quota, 0.25 * max(vol_1m_global, 1.0))

        # исполнение реверса
        if (self.reverse_quota > 0.0 and now >= self.reverse_lock_until and now >= self.reverse_earliest_ts):
            rb = min(self.reverse_quota, self._avg_sub_per_action_quota())
            if rb > 0.0:
                opp_side = self._dominant_opposite_side()
                out += self._compose_children(
                    snap, mid, rb, ofi, spread, opp_side,
                    allow_market=True, ttl_rng=(5, 15), ping_prob=0.2, mode="reverse"
                )
                self.reverse_quota = max(0.0, self.reverse_quota - rb)

        # цикл сабтасков
        for s in self.subs:
            # окна активности
            if now >= s.window_end_ts:
                self._roll_windows(s, now)
            if s.state == "idle":
                continue

            # локальные гейты
            allow_market = (now >= s.cool_off_until) and (now >= s.market_disabled_until)
            allow_market = allow_market and self._update_and_check_band(s, mid, now)

            # «усталость»
            if self.flow_energy > 35:
                if random.random() < 0.6:
                    allow_market = False
                s.next_action_after += random.uniform(5.0, 12.0)

            # тренд-гейт вероятностно
            if self._too_trendy(now, win=120):
                if random.random() < 0.6:
                    allow_market = False
                if random.random() < 0.3:
                    s.next_action_after += random.uniform(5.0, 12.0)
                if random.random() < 0.5:
                    self.reverse_quota += 0.06

            # капы
            r_cap = self._participation_cap(s, now)
            depth_cap = self._depth_cap(snap, s)
            base_rate = max(1e-6, min(r_cap * max(vol_1m_global, 1.0), depth_cap))

            # OU
            s.ou_x = max(0.2, min(2.0, ou_step(s.ou_x, s.profile.ou_theta, s.profile.ou_sigma)))

            # форма волны + факторы
            prog = 0.0
            if s.accum_target > 0:
                prog = min(1.0, s.accum_done / max(s.accum_target, 1e-9))
            shape_gain = {
                "front": 1.0 + 0.4 * prog,
                "back": 1.4 - 0.4 * prog,
                "middle": 1.0 - 0.8 * abs(0.5 - prog),
                "linear": 1.0
            }.get(s.shape, 1.0)
            s.rate_eps = 0.85 * s.rate_eps + random.gauss(0, 0.35)
            factor = 1.0 + 0.12 * (s.factor_load[0] * self.F1 + s.factor_load[1] * self.F2)

            desired_rate = base_rate * s.ou_x * shape_gain * max(0.6, min(1.6, 1 + 0.2 * s.rate_eps)) * factor

            # якорь VWAP + «база»
            dev = self._vwap_dev(order_book)
            accel_adj = 0.8 if dev > 0.6 * max(self.last_atr5m, 0.01) else 1.0
            if dev <= 0.3 * max(self.last_atr5m, 0.01) and abs(ofi) < 0.25 and s.base_mode_until < now:
                s.base_mode_until = now + random.randint(60, 120)
            if now < s.base_mode_until:
                allow_market = False

            # срочность
            rate = desired_rate * s.profile.accel * accel_adj * self._urgency_gain(s, now)

            # кулдаун
            if now - s.last_action_ts < s.next_action_after:
                continue
            s.last_action_ts = now
            s.next_action_after = self._next_cooldown(s.scale)

            per_action_qty = max(0.0, min(s.notional, rate * 0.25))
            if per_action_qty <= 0:
                continue

            # гамма-гейт слабее
            gm = self._gamma_mode(now)
            if gm == "long" and random.random() < 0.5:
                allow_market = False

            # анти-серийность
            if self.run_len >= 16:
                allow_market = False
                s.next_action_after += random.uniform(5.0, 10.0)
                self.reverse_quota += 0.06 * per_action_qty

            # ASD/волны
            self._ensure_accum_target(s, snap, vol_1m_global)
            thin = self._thin_book(snap, s.side)
            side_bias = ofi if s.side == OrderSide.BID else -ofi

            # блок на продолжение импульса до отката
            if s.sweep_block_until > now:
                allow_market = False

            # вход в sweep
            if (s.phase == "accumulate" and (thin or side_bias > self.regime.sweep_p + 0.02)
                    and s.accum_done >= 0.7 * s.accum_target and random.random() < self.regime.sweep_p):
                s.phase = "sweep"
                s.phase_until = now + random.randint(1, 3)
                s.sweep_budget = 0.15 * max(s.accum_target, 1.0)
                s.last_mid_on_sweep = mid
                s.post_sweep_check_ts = now + random.randint(20, 40)
                self.reverse_lock_until = max(self.reverse_lock_until, now + random.randint(6, 9))
                self.reverse_earliest_ts = now + random.randint(4, 7)
                s.wave_base_price = mid

            # завершение sweep по времени
            if s.phase == "sweep" and now >= s.phase_until:
                s.phase = "distribute"
                s.phase_until = now + random.randint(10, 20)

            # проверка ложного прорыва
            if s.post_sweep_check_ts and now >= s.post_sweep_check_ts:
                delta = abs(mid - s.last_mid_on_sweep)
                if delta < 0.3 * max(self.last_atr5m, 0.01) and random.random() < 0.5:
                    self.reverse_quota += 0.25 * max(s.accum_target, 1.0)
                s.post_sweep_check_ts = 0

            # быстрый импакт → ступень
            if s.phase == "sweep" and self._impact_too_fast(now):
                allow_market = False
                s.next_action_after += random.uniform(2.0, 5.0)

            # дистанция до края полосы
            edge_dist = min(abs(mid - s.band_edge), abs(s.band_edge + s.band_w - mid))

            # тактики
            if s.phase == "accumulate":
                out += self._compose_children(
                    snap, mid, per_action_qty, ofi, spread, s.side,
                    allow_market=False,
                    ttl_rng=(s.profile.ttl_min, s.profile.ttl_max),
                    ping_prob=0.6 * s.profile.ping_prob,
                    mode="acc"
                )
                s.accum_done += per_action_qty * 0.5
            elif s.phase == "sweep":
                qty = min(per_action_qty * random.uniform(0.6, 0.9), s.sweep_budget)
                s.sweep_budget = max(0.0, s.sweep_budget - qty)
                out += self._compose_children(
                    snap, mid, qty, ofi, spread, s.side,
                    allow_market=True, ttl_rng=(3, 6),
                    ping_prob=max(0.25, s.profile.ping_prob),
                    mode="sweep"
                )
                self.hour_stats["sweeps"] += 1
                self.reverse_quota += 0.15 * qty
                self.reverse_earliest_ts = max(self.reverse_earliest_ts, now + random.randint(4, 7))

                move = abs(mid - s.wave_base_price)
                if move >= 0.6 * max(self.last_atr5m, 0.01):
                    sign = 1 if s.side == OrderSide.BID else -1
                    s.sweep_block_until = now + random.randint(30, 60)
                    s.phase = "distribute"
                    s.phase_until = now + random.randint(10, 20)
                    s.wave_base_price = s.wave_base_price + sign * 0.35 * move
            else:  # distribute
                # у края полосы — микро-проба уровня
                if edge_dist < 0.15 * s.band_w and random.random() < 0.25:
                    probe = 0.30 * per_action_qty
                    out += self._compose_children(
                        snap, mid, probe, ofi, spread, s.side,
                        allow_market=True, ttl_rng=(2, 5), ping_prob=0.35, mode="sweep"
                    )
                    self.reverse_quota += 0.08 * probe

                out += self._compose_children(
                    snap, mid, per_action_qty, ofi, spread, s.side,
                    allow_market=allow_market,
                    ttl_rng=(s.profile.ttl_min, s.profile.ttl_max),
                    ping_prob=s.profile.ping_prob,
                    mode="dist"
                )
                self.reverse_quota += 0.06 * per_action_qty

                # цель отката выполнена?
                if s.wave_base_price != 0.0:
                    if (s.side == OrderSide.BID and mid <= s.wave_base_price) or (s.side == OrderSide.ASK and mid >= s.wave_base_price):
                        s.sweep_block_until = 0

                # межволновая логика
                if s.notional > 0 and s.waves_left > 1 and now >= s.phase_until:
                    s.waves_left -= 1
                    s.phase = "accumulate"
                    s.accum_done = 0.0
                    s.accum_target = 0.0
                    s.base_mode_until = 0
                    s.next_action_after += random.uniform(6.0, 12.0)
                elif s.notional <= 0 or s.waves_left <= 1:
                    if s.notional <= 0:
                        s.completed = True

        # лёгкий ММ
        if self.mm_enabled:
            out += self._market_make(snap, mid, ofi, now, vol_1m_global, macro_mode)

        return out

    # ---------- внутреннее ----------
    # mid/ATR
    def _update_mid_hist(self, mid: float, ts: int):
        if not self.mid_hist or self.mid_hist[-1][0] != ts:
            self.mid_hist.append((ts, mid))

    def _update_atrs(self, conn, now: int):
        try:
            h_c = load_candles(conn, "1h", now - 3600, now) if conn else []
            self.last_atr1h = atr_from_candles(h_c)
        except Exception:
            self.last_atr1h = max(self.last_atr1h, 0.01)
        try:
            c5 = load_candles(conn, "5m", now - 90 * 60, now) if conn else []
            self.last_atr5m = atr_from_candles(c5)
        except Exception:
            self.last_atr5m = max(self.last_atr5m, 0.01)

    # бюджеты
    def _budgets_ok(self) -> bool:
        if self.cum_executed <= 0:
            return True
        atr = max(self.last_atr1h, 0.01)
        avg_is = self.cum_slippage / max(self.cum_executed, 1e-9)
        return avg_is <= 0.8 * atr

    # режим часа
    def _update_regime(self, now: int):
        hr = (now // 3600)
        if hr == self.regime.last_hour:
            return
        prev_sweeps = self.hour_stats["sweeps"]
        prev_run = self.hour_stats["max_run"]
        prev_net = self.hour_stats["net_sign"]

        sweep_p = 0.15 + random.uniform(-0.03, 0.03)
        wave_mu = 2.0 + random.uniform(-0.4, 0.4)
        silent_rate = 12.0 + random.uniform(-3.0, 3.0)

        if prev_run >= 18 or abs(prev_net) > 60 or prev_sweeps > 20:
            sweep_p *= 0.75
            wave_mu += 0.4
            silent_rate += 6.0

        self.regime.sweep_p = max(0.08, min(0.28, sweep_p))
        self.regime.wave_count_mu = max(1.5, min(3.0, wave_mu))
        self.regime.silent_lambda_per_hour = max(6.0, min(20.0, silent_rate))
        self.regime.last_hour = hr
        self.hour_stats = {"sweeps": 0, "net_sign": 0, "max_run": 0}

    # окна
    def _maybe_macro_window(self, now: int):
        if now < self.macro_window_until:
            return
        if now < getattr(self.regime, "macro_refractory_until", 0):
            return
        h = (now // 3600) % 24
        bias = 1.6 if (7 <= h <= 11 or 12 <= h <= 16) else 1.0
        p = 0.00025 * bias  # ~раз в 60–120 мин
        if random.random() < p:
            dur = random.randint(8 * 60, 14 * 60)
            self.macro_window_until = now + dur
            self.regime.macro_cap_mult = random.uniform(1.10, 1.20)
            self.regime.macro_refractory_until = self.macro_window_until + random.randint(45 * 60, 75 * 60)
        else:
            self.regime.macro_cap_mult = 1.0

    def _maybe_start_silent_window(self, now: int):
        if self.silent_until > now:
            return
        lam = self.regime.silent_lambda_per_hour / 3600.0
        if random.random() < lam:
            self.silent_until = now + random.randint(90, 150)

    # активация сабтасков
    def _maybe_activate_subs(self, order_book, snap, now: int):
        n = len([s for s in self.subs if not s.completed])
        if n < self.min_subs or (n < self.max_subs and random.random() < 0.3):
            vol_1m = self._estimate_1m_volume(order_book)
            tob = 0.0
            if snap.get("bids"): tob += float(snap["bids"][0].get("volume", 0.0))
            if snap.get("asks"): tob += float(snap["asks"][0].get("volume", 0.0))
            ref = max(tob, 1.0)
            vol_mult = max(0.3, min(2.0, vol_1m / ref)) if vol_1m > 0 else 0.2
            h = (now // 3600) % 24
            tz = 1.4 if 7 <= h <= 11 else 1.6 if 12 <= h <= 16 else 0.7 if (20 <= h or h <= 3) else 1.0
            lam = 0.06 / 60.0 * vol_mult * tz
            if random.random() < 1.0 - math.exp(-lam):
                self._spawn_subtask(snap, now)
                if random.random() < 0.25 and n + 1 < self.max_subs:
                    self._spawn_subtask(snap, now)

    def _spawn_subtask(self, snap, now: int):
        mid = microprice_from_snapshot(snap)
        if mid is None:
            return
        gm = self._gamma_mode(now)
        p_buy = 0.7 if gm == "short" else (0.55 if gm == "long" else 0.6)

        # диверсификация стороны по недавнему знаку
        S = sum(self.sign_roll)
        if abs(S) > 80:
            p_buy = 0.25 if S > 0 else 0.75

        side = OrderSide.BID if random.random() < p_buy else OrderSide.ASK

        ln_mu, ln_sigma = math.log(80e6), 0.7
        notional = min(max(math.exp(random.gauss(ln_mu, ln_sigma)), 10e6), 200e6)

        r = random.random()
        if r < 0.33:
            scale = "micro"; horizon = random.randint(2 * 3600, 5 * 3600)
            prof = ExecProfile(0.02, 0.06, 0.12, 0.25, 0.80, 0.18, 3, 10, 1.0)
        elif r < 0.8:
            scale = "meso"; horizon = random.randint(6 * 3600, 18 * 3600)
            prof = ExecProfile(0.03, 0.08, 0.12, 0.20, 0.45, 0.22, 5, 18, 1.0)
        else:
            scale = "macro"; horizon = random.randint(2 * 3600, 6 * 3600)
            prof = ExecProfile(0.04, 0.09, 0.10, 0.15, 0.30, 0.18, 8, 25, 1.1)

        deadline_ts = None
        if random.random() < 0.25:
            tm = time.localtime(now)
            target = time.mktime((tm.tm_year, tm.tm_mon, tm.tm_mday, 16, 0, 0, tm.tm_wday, tm.tm_yday, tm.tm_isdst))
            if target <= now:
                target += 24 * 3600
            deadline_ts = int(target)

        waves = max(1, int(random.gauss(self.regime.wave_count_mu, 0.6)))

        s = SubTask(
            sub_id=str(uuid.uuid4()), side=side, notional=notional, started_at=now, horizon_sec=horizon,
            deadline_ts=deadline_ts, arrival_price=mid, profile=prof, scale=scale, waves_left=waves
        )
        s.factor_load = (random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7))
        s.shape = random.choice(["front", "back", "middle", "linear"])

        self._roll_windows(s, now)
        self._init_band(s, mid, now)
        self.subs.append(s)

    # окна burst/idle
    def _roll_windows(self, s: SubTask, now: int):
        if s.state == "idle":
            s.state = "burst"
            s.window_end_ts = now + self._draw_lognorm(3.7, 0.9, 20, 140)
        else:
            if random.random() < 0.5:
                s.state = "burst"
                s.window_end_ts = now + random.randint(8, 15)
            else:
                s.state = "idle"
                s.window_end_ts = now + self._draw_lognorm(4.1, 0.9, 45, 210)

    # полосы
    def _init_band(self, s: SubTask, mid: float, now: int):
        bw = max(0.1 * self.last_atr1h, 0.01)
        bw = max(bw, 0.6 * max(self.last_atr5m, 0.01))
        gm = self._gamma_mode(now)
        trend = (gm == "short" and s.side == OrderSide.BID) or (gm == "long" and s.side == OrderSide.ASK)
        if trend:
            bw *= 1.3
        s.band_w = bw
        s.band_edge = math.floor(mid / bw) * bw

    def _update_and_check_band(self, s: SubTask, mid: float, now: int) -> bool:
        if s.band_w <= 0.0 or s.band_edge is None:
            self._init_band(s, mid, now)
        moved = False
        while True:
            if s.side == OrderSide.BID and mid > s.band_edge + s.band_w:
                s.band_edge += s.band_w; moved = True
            elif s.side == OrderSide.ASK and mid < s.band_edge:
                s.band_edge -= s.band_w; moved = True
            else:
                break
        if moved:
            if random.random() < 0.35:
                s.last_action_ts = now - int(random.uniform(0.0, 0.8))
                s.next_action_after = random.uniform(8.0, 20.0)
            if s.state == "burst" and s.phase == "sweep":
                s.next_action_after += random.uniform(6.0, 10.0)
        # ловушка на 30м экстремумах
        if self._thirty_min_break_trap(mid):
            self.reverse_quota += 0.25 * max(s.accum_target, 1.0)
            self.reverse_earliest_ts = max(self.reverse_earliest_ts, now + random.randint(4, 7))
            self.reverse_lock_until = now
        return True

    # капы и объёмы
    def _participation_cap(self, s: SubTask, now: int) -> float:
        h = (now // 3600) % 24
        season = 1.3 if 7 <= h <= 11 else 1.5 if 12 <= h <= 16 else 0.7 if (20 <= h or h <= 3) else 1.0
        lo, hi = s.profile.r_lo, s.profile.r_hi * season * self.regime.macro_cap_mult
        return max(0.0, min(hi, max(lo, lo + (hi - lo) * random.random())))

    def _depth_cap(self, snap, s: SubTask) -> float:
        opp = OrderSide.ASK if s.side == OrderSide.BID else OrderSide.BID
        d10 = sum_depth(snap, opp, 10)
        return max(0.0, s.profile.lob_beta * d10)

    def _estimate_1m_volume(self, order_book) -> float:
        th = getattr(order_book, "trade_history", None)
        if th is None:
            return 0.0
        tnow = time.time()
        vol = 0.0
        cnt = 0
        for tr in reversed(th):
            ts = tr.get("timestamp", tnow)
            if tnow - ts > 60.0:
                break
            vol += float(tr.get("volume", 0.0))
            cnt += 1
            if cnt >= 800:
                break
        if vol <= 0.0:
            snap = order_book.get_order_book_snapshot(depth=1)
            bvol = snap.get("bids", [{}])[0].get("volume", 0.0) if snap.get("bids") else 0.0
            avol = snap.get("asks", [{}])[0].get("volume", 0.0) if snap.get("asks") else 0.0
            vol = 0.5 * (bvol + avol)
            # U-профиль дня и шум
            tm = time.localtime(tnow)
            h = tm.tm_hour + tm.tm_min / 60.0
            u = 0.6 + 0.4 * math.cos((h - 12) / 12 * math.pi)
            vol *= random.uniform(0.7, 1.3) * u
        return max(vol, 0.0)

    # режим
    def _gamma_mode(self, now: int) -> str:
        if not self.mid_hist:
            return "neutral"
        m_now = self.mid_hist[-1][1]
        ref_ts = now - 10 * 60
        ref = None
        for ts, m in reversed(self.mid_hist):
            if ts <= ref_ts:
                ref = m
                break
        if ref is None:
            ref = self.mid_hist[0][1]
        move = abs(m_now - ref)
        th = max(0.5 * self.last_atr5m, 0.01)
        return "long" if move < th else "short"

    # анти-трейс: охлаждаем часть доминирующей стороны
    def _anti_trace_controller(self, now: int):
        series = [sgn for ts, sgn in self.sign_hist if now - ts <= 300]
        n = len(series)
        if n >= 20:
            mean = sum(series) / n
            var = sum((x - mean) ** 2 for x in series) / max(n - 1, 1)
            rho1 = 1.0 if var <= 1e-9 else sum((series[i] - mean) * (series[i - 1] - mean) for i in range(1, n)) / max(n - 1, 1) / var
            imbalance = abs(sum(series)) / max(n, 1)
            self.hour_stats["max_run"] = max(self.hour_stats["max_run"], self.run_len)
            self.hour_stats["net_sign"] = sum(series)
            if rho1 > 0.65 or imbalance > 0.70:
                dom = OrderSide.BID if sum(series) > 0 else OrderSide.ASK
                targets = [x for x in self.subs if x.side == dom and not x.completed]
                if targets:
                    k = max(1, len(targets) // 2)
                    for s in random.sample(targets, k=k):
                        s.cool_off_until = max(s.cool_off_until, now + random.randint(10, 40))
                        s.market_disabled_until = max(s.market_disabled_until, now + random.randint(5, 20))

    # срочность
    def _urgency_gain(self, s: SubTask, now: int) -> float:
        if s.deadline_ts:
            left = max(0, s.deadline_ts - now); total = max(2 * 3600, s.horizon_sec)
        else:
            left = max(0, (s.started_at + s.horizon_sec) - now); total = max(s.horizon_sec, 1)
        progress = 1.0 - left / max(total, 1)
        base = 1.0 + (0.5 if s.scale == "micro" else 0.9 if s.scale == "macro" else 0.7) * progress
        return max(0.6, min(2.0, base))

    # OFI
    def _compute_ofi(self, snap) -> Tuple[float, float]:
        pb, vb, pa, va = top_of_book(snap)
        spread = (pa - pb) if (pb == pb and pa == pa) else float("inf")
        if self.prev_pb is None:
            ofi = 0.0
        else:
            d_bid = (vb if pb >= self.prev_pb else -self.prev_vb) if self.prev_vb is not None else 0.0
            d_ask = (va if pa <= self.prev_pa else -self.prev_va) if self.prev_va is not None else 0.0
            ofi = (d_bid - d_ask) / max((vb + va), 1e-9)
            ofi = max(-1.5, min(1.5, ofi))
        self.prev_pb, self.prev_vb, self.prev_pa, self.prev_va = pb, vb, pa, va
        return ofi, spread

    # дети
    def _compose_children(self, snap, mid: float, qty: float, ofi: float, spread: float, side: OrderSide,
                          allow_market: bool = True, ttl_rng: Tuple[int, int] = (5, 20), ping_prob: float = 0.2,
                          mode: str = "normal") -> List[Order]:
        res: List[Order] = []
        my = snap.get("bids" if side == OrderSide.BID else "asks", [])
        opp = snap.get("asks" if side == OrderSide.BID else "bids", [])
        if not opp:
            return res
        best_opp = opp[0]["price"]
        best_my = my[0]["price"] if my else (best_opp - self.default_tick if side == OrderSide.BID else best_opp + self.default_tick)

        # тонкость
        opp_s = OrderSide.ASK if side == OrderSide.BID else OrderSide.BID
        d3 = sum_depth(snap, opp_s, 3)
        d10 = sum_depth(snap, opp_s, 10)
        thin = (d10 > 0 and d3 / d10 < 0.25)

        # число срезов (1..5) и веса stick-breaking
        lam = 2.0
        n_slices = max(1, min(5, int(random.expovariate(1 / lam)) + 1))
        weights, remain = [], 1.0
        for i in range(n_slices - 1):
            cut = random.betavariate(1.0, 1.0 + 0.8 * i)
            w = remain * cut
            weights.append(w)
            remain -= w
        weights.append(remain)
        random.shuffle(weights)

        # уровни айсберга 1..3
        levels = 1 if random.random() < 0.6 else (2 if random.random() < 0.85 else 3)

        for w in weights:
            slice_qty = max(qty * w * random.uniform(0.8, 1.25), 1.0)
            ttl = random.randint(ttl_rng[0], ttl_rng[1])
            tight = spread <= 2 * self.default_tick
            bias = ofi if side == OrderSide.BID else -ofi

            # тактика
            r = random.random()
            if r < 0.4 and mode != "sweep":
                tactic = "passive_spread"
            elif r < 0.8 and mode != "sweep":
                tactic = "join_best"
            else:
                tactic = "market" if allow_market and (thin or bias > 0.1 or tight or mode == "sweep") and random.random() < max(0.15, ping_prob) else "join_best"

            if tactic in ("passive_spread", "join_best"):
                for lvl in range(levels):
                    step = (lvl + 1) * self.default_tick * random.randint(1, 2)
                    px = (best_my - step) if side == OrderSide.BID else (best_my + step)
                    oid = str(uuid.uuid4())
                    res.append(Order(order_id=oid, agent_id=self.agent_id, side=side,
                                     volume=slice_qty / levels, price=quant(px),
                                     order_type=OrderType.LIMIT, ttl=ttl + lvl))
                    self.order_owner[oid] = self._current_sub_id_or_default(mode, side)
                    self._register_sign(now_ts(), +1 if side == OrderSide.BID else -1)
            else:
                m_vol = max(min(slice_qty * random.uniform(0.4, 0.9), 0.8 * slice_qty), 1.0)
                oid = str(uuid.uuid4())
                res.append(Order(order_id=oid, agent_id=self.agent_id, side=side,
                                 volume=m_vol, price=None, order_type=OrderType.MARKET, ttl=None))
                self.order_owner[oid] = self._current_sub_id_or_default(mode, side)
                self._register_sign(now_ts(), +1 if side == OrderSide.BID else -1)

        # небольшой встречный хвост
        if mode in ("normal", "dist") and random.random() < 0.12:
            opp_side = OrderSide.ASK if side == OrderSide.BID else OrderSide.BID
            tiny = max(qty * random.uniform(0.04, 0.1), 1.0)
            oid = str(uuid.uuid4())
            res.append(Order(order_id=oid, agent_id=self.agent_id, side=opp_side, volume=tiny,
                             price=quant(best_my), order_type=OrderType.LIMIT,
                             ttl=random.randint(ttl_rng[0], ttl_rng[1])))
            self.order_owner[oid] = self._current_sub_id_or_default(mode, opp_side)
            self._register_sign(now_ts(), +1 if opp_side == OrderSide.BID else -1)

        return res

    # ASD helpers
    def _ensure_accum_target(self, s: SubTask, snap, vol_1m: float):
        if s.accum_target <= 0.0 or (s.phase == "accumulate" and s.accum_done >= s.accum_target):
            opp = OrderSide.ASK if s.side == OrderSide.BID else OrderSide.BID
            d10 = sum_depth(snap, opp, 10)
            s.accum_target = max(1.0, min(0.3 * d10, 0.15 * max(vol_1m, 1.0)))
            s.accum_done = 0.0
            s.phase = "accumulate"
            s.phase_until = 0
            s.sweep_budget = 0.0

    def _thin_book(self, snap, side: OrderSide) -> bool:
        opp = OrderSide.ASK if side == OrderSide.BID else OrderSide.BID
        d3 = sum_depth(snap, opp, 3)
        d10 = sum_depth(snap, opp, 10)
        return d10 > 0 and d3 / d10 < 0.25

    # sign utils
    def _current_sub_id_or_default(self, mode: str, side: OrderSide) -> str:
        for s in self.subs:
            if not s.completed and s.side == side:
                return s.sub_id
        return "manager"

    def _register_sign(self, ts: int, sign: int):
        sgn = 1 if sign >= 0 else -1
        self.sign_hist.append((ts, sgn))
        self.sign_roll.append(sgn)
        self.run_len = self.run_len + 1 if sgn == self.prev_sign else 1
        self.prev_sign = sgn
        self.flow_energy = 0.98 * self.flow_energy + abs(sgn)

    def _dominant_opposite_side(self) -> OrderSide:
        s = sum(sgn for _, sgn in self.sign_hist)
        return OrderSide.ASK if s > 0 else OrderSide.BID

    def _avg_sub_per_action_quota(self) -> float:
        act = [s for s in self.subs if not s.completed]
        if not act:
            return 1.0
        avg_hi = sum(s.profile.r_hi for s in act) / len(act)
        return max(1.0, 0.25 * avg_hi * max(self._last_vol_1m, 1.0))

    def _next_cooldown(self, scale: str) -> float:
        return random.uniform(0.5, 1.5) if scale == "micro" else random.uniform(2.0, 4.0) if scale == "macro" else random.uniform(1.0, 3.0)

    def _draw_lognorm(self, mu: float, sigma: float, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(random.lognormvariate(mu, sigma))))

    # импакт-лимитер
    def _impact_too_fast(self, now: int) -> bool:
        if len(self.mid_hist) < 3:
            return False
        (t0, m0) = self.mid_hist[-3]
        (t2, m2) = self.mid_hist[-1]
        dt = max(1, t2 - t0)
        v = abs(m2 - m0) / dt
        return v > 0.06 * max(self.last_atr5m, 0.01)

    # VWAP якорь
    def _vwap_dev(self, order_book) -> float:
        th = getattr(order_book, "trade_history", None) or []
        tnow = time.time()
        pv = v = 0.0
        for tr in reversed(th):
            if tnow - tr.get("timestamp", tnow) > 900:
                break
            p = float(tr.get("price", 0.0))
            q = float(tr.get("volume", 0.0))
            pv += p * q
            v += q
        if v <= 0:
            return 0.0
        vwap = pv / v
        mid = self.mid_hist[-1][1] if self.mid_hist else vwap
        return abs(mid - vwap)

    # 30m false-break ловушка
    def _thirty_min_break_trap(self, mid: float) -> bool:
        tnow = now_ts()
        xs = [m for t, m in self.mid_hist if t >= tnow - 1800]
        if len(xs) < 5:
            return False
        lo, hi = min(xs), max(xs)
        thr = 0.15 * max(self.last_atr5m, 0.01)
        broke_hi = (mid > hi) and (mid - hi) <= thr
        broke_lo = (mid < lo) and (lo - mid) <= thr
        return random.random() < 0.6 if (broke_hi or broke_lo) else False

    # тренд-гейт
    def _too_trendy(self, now: int, win: int = 60) -> bool:
        xs = [m for t, m in self.mid_hist if now - t <= win]
        if len(xs) < 3:
            return False
        drift = abs(xs[-1] - xs[0])
        return drift > 0.9 * max(self.last_atr5m, 0.01)

    # ---------- ММ ----------
    def _market_make(self, snap, mid, ofi, now: int, vol_1m: float, macro_mode: bool):
        if now < self.silent_until or now < self.mm_cooldown_until:
            return []
        pb, vb, pa, va = top_of_book(snap)
        if not (pb == pb and pa == pa):
            return []
        spread = pa - pb
        if spread < self.default_tick:
            return []

        dBid10 = sum_depth(snap, OrderSide.BID, 10)
        dAsk10 = sum_depth(snap, OrderSide.ASK, 10)
        if dBid10 + dAsk10 <= 0:
            return []
        imb_depth = (dBid10 - dAsk10) / (dBid10 + dAsk10)

        base_spread = max(self.mm_min_spread, 0.35 * max(self.last_atr5m, 0.01))

        inv_skew = max(-1.0, min(1.0, self.inv / max(self.inv_max, 1.0)))
        flow_skew = max(-1.0, min(1.0, ofi))
        depth_skew = max(-1.0, min(1.0, imb_depth))
        skew = 0.5 * inv_skew + 0.3 * flow_skew + 0.2 * depth_skew
        skew_mid = mid + skew * 0.3 * max(self.last_atr5m, 0.01)

        cap_by_vol = 0.08 * max(vol_1m, 1.0)
        cap_by_depth = 0.10 * min(dBid10, dAsk10)
        q_base = max(1.0, min(cap_by_vol, cap_by_depth))
        if macro_mode:
            q_base *= 0.35

        ofi_mult = 1.0
        if abs(ofi) > 0.8:
            return []
        elif abs(ofi) > 0.6:
            ofi_mult = 0.15
        elif abs(ofi) > 0.4:
            ofi_mult = 0.3
        q_base *= ofi_mult

        q_bid = q_base * (1.0 - max(0.0, self.inv / max(self.inv_max, 1.0)))
        q_ask = q_base * (1.0 - max(0.0, -self.inv / max(self.inv_max, 1.0)))
        if q_bid < 1.0 and q_ask < 1.0:
            if self.inv > 0:
                q_bid = 0.0; q_ask = max(1.0, q_base * 0.6)
            else:
                q_ask = 0.0; q_bid = max(1.0, q_base * 0.6)

        bid_px = quant(min(pa - self.default_tick, skew_mid - base_spread / 2.0))
        ask_px = quant(max(pb + self.default_tick, skew_mid + base_spread / 2.0))
        if bid_px >= ask_px:
            mid0 = (pb + pa) / 2.0
            bid_px = quant(mid0 - self.default_tick)
            ask_px = quant(mid0 + self.default_tick)
            if bid_px >= ask_px:
                return []

        ttl = random.randint(2, 6)
        orders = []
        if q_bid >= 1.0:
            oid = str(uuid.uuid4())
            orders.append(Order(order_id=oid, agent_id=self.agent_id, side=OrderSide.BID,
                                volume=q_bid, price=bid_px, order_type=OrderType.LIMIT, ttl=ttl))
            self.order_owner[oid] = "mm"
            self._register_sign(now, +1)
        if q_ask >= 1.0:
            oid = str(uuid.uuid4())
            orders.append(Order(order_id=oid, agent_id=self.agent_id, side=OrderSide.ASK,
                                volume=q_ask, price=ask_px, order_type=OrderType.LIMIT, ttl=ttl))
            self.order_owner[oid] = "mm"
            self._register_sign(now, -1)

        self.mm_cooldown_until = now + random.randint(1, 3)
        return orders
