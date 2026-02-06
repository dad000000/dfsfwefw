from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


class OnlineStandardizer:
    """Online feature standardizer (Welford algorithm)."""

    def __init__(self, dim: int, eps: float = 1e-9) -> None:
        self.dim = int(dim)
        self.eps = float(eps)
        self.n: int = 0
        self.mean: List[float] = [0.0] * self.dim
        self.m2: List[float] = [0.0] * self.dim

    def update(self, x: List[float]) -> None:
        if len(x) != self.dim:
            return
        self.n += 1
        n = float(self.n)
        for i in range(self.dim):
            xi = float(x[i])
            delta = xi - self.mean[i]
            self.mean[i] += delta / n
            delta2 = xi - self.mean[i]
            self.m2[i] += delta * delta2

    def transform(self, x: List[float]) -> List[float]:
        if len(x) != self.dim:
            return [0.0] * self.dim
        if self.n < 2:
            return [float(v) for v in x]
        out: List[float] = [0.0] * self.dim
        denom = max(float(self.n - 1), 1.0)
        for i in range(self.dim):
            var = self.m2[i] / denom
            std = math.sqrt(max(var, self.eps))
            out[i] = (float(x[i]) - self.mean[i]) / std
        return out

    def update_and_transform(self, x: List[float]) -> List[float]:
        self.update(x)
        return self.transform(x)


class OnlineLogReg:
    """Tiny online logistic regression (SGD) with L2 regularization."""

    def __init__(self, dim: int, lr: float = 0.05, l2: float = 1e-4) -> None:
        self.dim = int(dim)
        self.lr = float(lr)
        self.l2 = float(l2)
        # weights include bias as last element
        self.w: List[float] = [0.0] * (self.dim + 1)

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict_proba(self, x: List[float]) -> float:
        if len(x) != self.dim:
            return 0.5
        z = self.w[-1]
        for i in range(self.dim):
            z += self.w[i] * float(x[i])
        return float(self._sigmoid(z))

    def update(self, x: List[float], y: int) -> float:
        p = self.predict_proba(x)
        err = p - float(y)
        lr = float(self.lr)
        l2 = float(self.l2)
        for i in range(self.dim):
            grad = err * float(x[i]) + l2 * self.w[i]
            self.w[i] -= lr * grad
        self.w[-1] -= lr * err
        return float(p)


@dataclass
class MarketProbe:
    ts_ms: int
    entry_mid: float
    x_raw: List[float]
    done_mask: int = 0  # bit i -> horizon i trained


class MicroMLPredictor:
    """Online micro-ML predictor for mid move direction at multiple horizons.

    - Collects probes at a fixed interval.
    - Trains a separate online log-reg model per horizon (200/500/1000ms).
    """

    def __init__(
        self,
        horizons_ms: Tuple[int, ...] = (200, 500, 1000),
        feature_dim: int = 8,
        lr: float = 0.05,
        l2: float = 1e-4,
        max_probes: int = 5000,
    ) -> None:
        self.horizons_ms: Tuple[int, ...] = tuple(int(h) for h in horizons_ms)
        self.feature_dim = int(feature_dim)
        self.scalers: Dict[int, OnlineStandardizer] = {h: OnlineStandardizer(self.feature_dim) for h in self.horizons_ms}
        self.models: Dict[int, OnlineLogReg] = {h: OnlineLogReg(self.feature_dim, lr=lr, l2=l2) for h in self.horizons_ms}
        self.n_samples: Dict[int, int] = {h: 0 for h in self.horizons_ms}
        self.probes: Deque[MarketProbe] = deque(maxlen=int(max_probes))

    def observe(self, ts_ms: int, mid: float, x_raw: List[float]) -> None:
        if mid <= 0:
            return
        if len(x_raw) != self.feature_dim:
            return
        self.probes.append(MarketProbe(ts_ms=int(ts_ms), entry_mid=float(mid), x_raw=[float(v) for v in x_raw], done_mask=0))

    def update_with_current_mid(self, now_ms: int, mid_now: float) -> None:
        """Resolve matured probes with the current mid, training each horizon once."""
        if not self.probes or mid_now <= 0:
            return

        now_ms = int(now_ms)
        mid_now = float(mid_now)
        max_h = max(self.horizons_ms)
        full_mask = (1 << len(self.horizons_ms)) - 1

        keep: Deque[MarketProbe] = deque(maxlen=self.probes.maxlen)
        for p in list(self.probes):
            age = now_ms - int(p.ts_ms)
            if age < 0:
                continue
            # Drop very old probes
            if age > (max_h + 60_000):
                continue

            done_mask = int(p.done_mask)
            for i, h in enumerate(self.horizons_ms):
                bit = 1 << i
                if done_mask & bit:
                    continue
                if age >= int(h):
                    y = 1 if mid_now > float(p.entry_mid) else 0
                    scaler = self.scalers[int(h)]
                    model = self.models[int(h)]
                    x = scaler.update_and_transform(p.x_raw)
                    model.update(x, y)
                    self.n_samples[int(h)] += 1
                    done_mask |= bit

            if done_mask != full_mask:
                p.done_mask = done_mask
                keep.append(p)

        self.probes = keep

    def predict(self, x_raw: List[float], horizon_ms: int) -> float:
        h = int(horizon_ms)
        if h not in self.models:
            return 0.5
        if len(x_raw) != self.feature_dim:
            return 0.5
        x = self.scalers[h].transform([float(v) for v in x_raw])
        return float(self.models[h].predict_proba(x))

    def best_signal(self, x_raw: List[float], prefer_h_ms: int, min_n: int) -> Tuple[int, float, float, int]:
        """Return (h, p_up, edge, n) choosing a horizon with enough samples.

        - Prefer prefer_h_ms if it has >= min_n samples.
        - Else pick the horizon with the most samples.
        """
        prefer_h = int(prefer_h_ms)
        min_n = int(min_n)
        if prefer_h in self.horizons_ms and int(self.n_samples.get(prefer_h, 0)) >= min_n:
            h = prefer_h
        else:
            # choose horizon with most samples
            h = max(self.horizons_ms, key=lambda hh: int(self.n_samples.get(int(hh), 0)))
        n = int(self.n_samples.get(int(h), 0))
        p = float(self.predict(x_raw, int(h)))
        edge = abs(p - 0.5)
        return int(h), p, float(edge), int(n)


@dataclass(frozen=True)
class BanditArm:
    name: str
    base_inside_ticks: int
    signal_skew_ticks: int
    requote_price_ticks: int
    requote_min_ms: int
    qty_mult: float
    min_edge_mult: float = 1.0


class EpsilonGreedyBandit:
    """Epsilon-greedy bandit over a small set of quoting arms."""

    def __init__(
        self,
        arms: List[BanditArm],
        epsilon: float = 0.20,
        # Compatibility: strategy.py passes epsilon_min
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        ewma_alpha: float = 0.15,
        switch_ms: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        if not arms:
            raise ValueError("Bandit requires at least one arm")
        self.arms = list(arms)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.alpha = float(ewma_alpha)
        self.switch_ms = int(switch_ms)
        if seed is not None:
            random.seed(int(seed))

        self.values: Dict[str, float] = {a.name: 0.0 for a in self.arms}
        self.counts: Dict[str, int] = {a.name: 0 for a in self.arms}

        self.current: BanditArm = self.arms[0]
        self.last_switch_ms: int = 0

    def _best_arm(self) -> BanditArm:
        best_val = None
        best: List[BanditArm] = []
        for a in self.arms:
            v = float(self.values.get(a.name, 0.0))
            if best_val is None or v > best_val:
                best_val = v
                best = [a]
            elif v == best_val:
                best.append(a)
        return random.choice(best) if best else self.current

    def maybe_switch(self, now_ms: int, reward: float) -> Tuple[BanditArm, bool]:
        """Update value for current arm and maybe switch.

        Returns: (arm, switched)
        """
        now_ms = int(now_ms)
        if self.last_switch_ms == 0:
            self.last_switch_ms = now_ms
            return self.current, True

        if (now_ms - int(self.last_switch_ms)) < int(self.switch_ms):
            return self.current, False

        # update value for current arm
        name = self.current.name
        prev = float(self.values.get(name, 0.0))
        a = float(self.alpha)
        self.values[name] = (1.0 - a) * prev + a * float(reward)
        self.counts[name] = int(self.counts.get(name, 0)) + 1

        # choose next arm
        if random.random() < float(self.epsilon):
            self.current = random.choice(self.arms)
        else:
            self.current = self._best_arm()

        # decay epsilon
        self.epsilon = max(float(self.epsilon_min), float(self.epsilon) * float(self.epsilon_decay))
        self.last_switch_ms = now_ms
        return self.current, True

