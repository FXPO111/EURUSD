import numpy as np
from collections import deque
import random
import math
from dataclasses import dataclass


@dataclass
class MarketPhase:
    name: str
    confidence: float
    reason: dict


class MarketContextAdvanced:
    def __init__(self, maxlen_short=100, maxlen_medium=500, maxlen_long=2000):
        self.mid_prices_short = deque(maxlen=maxlen_short)
        self.mid_prices_medium = deque(maxlen=maxlen_medium)
        self.mid_prices_long = deque(maxlen=maxlen_long)

        self.snapshots = deque(maxlen=maxlen_long)
        self.sweep_events = deque(maxlen=maxlen_medium)

        self.phase = MarketPhase("undefined", 0.0, {})
        self.previous_phase_name = None
        self.phase_persistence = 0

        self.micro_trends = deque(maxlen=200)
        self.cluster_imbalances = deque(maxlen=200)
        self.volatility_levels = deque(maxlen=200)
        self.structural_swings = deque(maxlen=200)

        self.market_memory_trace = deque(maxlen=10)  # Память рыночных дней

        self.current_utc_time = 0
        self.current_session = "asia"
        self.session_progress_ratio = 0.0

        self.bank_flow_pressure = 0.0
        self.institutional_bias = 0.0
        self.macro_bias_vector = 0.0

        self.volatility_regime = "normal"
        self.session_type = "average"
        self.rollover_flag = False

        self.exhaustion_signal = False
        self.sweep_detected = False
        self.cluster_disbalance = 0.0

        self.daily_macro_scenario = random.choice([
            "neutral_day", "fed_expectation", "ecb_dovish", "eur_surge", "usd_liquidity_drain"
        ])
        self.session_type_profile = {
            "asia": random.choice(["quiet", "volatile", "spiky"]),
            "london": random.choice(["trending", "reversal", "balanced"]),
            "newyork": random.choice(["reactive", "continuation", "range"])
        }
        self.volatility_amplitude = random.uniform(0.8, 1.5)

        self.daily_phase_shift = random.uniform(0, 2 * math.pi)
        self.volatility_noise_factor = np.random.normal(1.0, 0.2)
        self.current_activity_level = 1.0

        self.demand_supply_skew = 0.0
        self.sticky_price_zone = None

        self.event_shock_level = 0.0

    def update(self, tick=None, mid_price=None, snapshot=None, sweep=None):
        if tick is not None:
            self.current_utc_time = tick % (24 * 60 * 60)

        if mid_price is not None:
            self._update_price_memory(mid_price)

        if snapshot is not None:
            self._update_cluster_signals(snapshot)

        self._update_sessions()
        self._update_volatility()
        self._update_macro_micro_logic()
        self._update_activity_cycle(tick or 0)
        self._update_phase()
        self._update_memory_trace()

    def _update_sessions(self):
        utc_hour = (self.current_utc_time // 3600) % 24
        if 0 <= utc_hour < 6:
            self.current_session = "asia"
        elif 6 <= utc_hour < 13:
            self.current_session = "london"
        else:
            self.current_session = "newyork"

        self.session_progress_ratio = (self.current_utc_time % 3600) / 3600.0
        self.rollover_flag = 22 <= utc_hour <= 23

        if self.current_session == "london" and self.session_type_profile['london'] == "trending":
            self.institutional_bias += np.random.normal(0.001, 0.0005)
        elif self.current_session == "newyork" and self.session_type_profile['newyork'] == "reversal":
            self.institutional_bias -= np.random.normal(0.001, 0.0005)

    def _update_price_memory(self, price):
        self.mid_prices_short.append(price)
        self.mid_prices_medium.append(price)
        self.mid_prices_long.append(price)

        if len(self.mid_prices_short) > 10:
            delta = self.mid_prices_short[-1] - self.mid_prices_short[0]
            self.micro_trends.append(delta)

    def _update_volatility(self):
        if len(self.mid_prices_short) >= 10:
            volatility = np.std(list(self.mid_prices_short)) * self.volatility_amplitude * self.volatility_noise_factor
            self.volatility_levels.append(volatility)

            if volatility < 0.01:
                self.volatility_regime = "low"
            elif volatility < 0.05:
                self.volatility_regime = "normal"
            else:
                self.volatility_regime = "high"

    def _update_cluster_signals(self, snapshot):
        if not snapshot:
            return

        best_bid = max(snapshot['bids'].keys()) if snapshot['bids'] else None
        best_ask = min(snapshot['asks'].keys()) if snapshot['asks'] else None

        if best_bid and best_ask:
            bid_volume = snapshot['bids'][best_bid]
            ask_volume = snapshot['asks'][best_ask]
            imbalance = (bid_volume - ask_volume) / max(bid_volume + ask_volume, 1e-6)
            self.cluster_imbalances.append(imbalance)
            self.cluster_disbalance = imbalance

            if abs(imbalance) > 0.7:
                self.sweep_detected = True

    def _update_activity_cycle(self, tick):
        activity_cycle = math.sin((tick / 3600) * math.pi * 2 + self.daily_phase_shift)
        self.current_activity_level = np.clip(activity_cycle * self.volatility_amplitude, 0.2, 2.0)

    def _update_phase(self):
        trend = np.mean(self.micro_trends) if self.micro_trends else 0.0
        vol = np.mean(self.volatility_levels) if self.volatility_levels else 0.0

        transition_probs = {
            "flat": {"trend_up": 0.1, "trend_down": 0.1, "volatile": 0.2},
            "trend_up": {"volatile": 0.3, "flat": 0.1},
            "trend_down": {"volatile": 0.3, "flat": 0.1},
            "volatile": {"flat": 0.4, "trend_up": 0.2, "trend_down": 0.2}
        }

        current = self.phase.name if self.phase.name in transition_probs else "flat"
        next_phase = current

        if vol < 0.01:
            next_phase = "flat"
        elif abs(trend) > 0.05:
            next_phase = "trend_up" if trend > 0 else "trend_down"
        else:
            next_phase = "volatile"

        if random.random() < transition_probs.get(current, {}).get(next_phase, 0.2):
            self.phase = MarketPhase(next_phase, min(1.0, abs(trend) * 20), {
                "volatility": vol,
                "trend": trend,
                "cluster_disbalance": self.cluster_disbalance
            })

    def _update_macro_micro_logic(self):
        bias = 0.0

        if self.daily_macro_scenario == "fed_expectation":
            bias -= 0.002
            if self.session_progress_ratio > 0.8:
                self.event_shock_level += 0.01
        elif self.daily_macro_scenario == "ecb_dovish":
            bias += 0.002

        if self.session_type_profile[self.current_session] == "spiky":
            bias += random.uniform(-0.0015, 0.0015)

        self.macro_bias_vector = bias + self.event_shock_level
        self.demand_supply_skew = (self.cluster_disbalance +
                                   self.institutional_bias +
                                   self.macro_bias_vector) * self.current_activity_level

        if self.rollover_flag:
            self.demand_supply_skew *= 0.3

    def _update_memory_trace(self):
        if self.current_utc_time % (60 * 60 * 3) == 0:
            trace = {
                "volatility": np.mean(self.volatility_levels) if self.volatility_levels else 0.0,
                "trend": np.mean(self.micro_trends) if self.micro_trends else 0.0,
                "skew": np.mean(self.cluster_imbalances) if self.cluster_imbalances else 0.0
            }
            self.market_memory_trace.append(trace)

    def get_session(self):
        return self.current_session

    def get_volatility_boost(self):
        return {
            "low": 0.7,
            "normal": 1.0,
            "high": 1.4
        }.get(self.volatility_regime, 1.0)

    def get_macro_bias(self):
        return self.macro_bias_vector

    def get_demand_supply_skew(self):
        return self.demand_supply_skew

    def get_liquidity_state(self):
        if abs(self.cluster_disbalance) > 0.8:
            return "toxic"
        elif self.volatility_regime == "low":
            return "stable"
        elif self.volatility_regime == "high":
            return "shaky"
        return "normal"

    def is_fixing_time(self):
        minute = (self.current_utc_time // 60) % 60
        hour = (self.current_utc_time // 3600) % 24
        return (hour == 9 and 30 <= minute < 45) or (hour == 15 and 30 <= minute < 45)

    def get_pressure_map(self):
        return {
            "macro": self.macro_bias_vector,
            "institutional": self.institutional_bias,
            "cluster": self.cluster_disbalance
        }

    def update(self, tick=None, mid_price=None, snapshot=None, sweep=None):
        if tick is not None:
            self.last_tick = tick

    def is_ready(self) -> bool:
        """
        Контекст считается готовым, если накоплено достаточно данных
        для вычисления фазы рынка и волатильности.
        """
        return len(self.mid_prices_short) >= 10 and len(self.volatility_levels) >= 5

    