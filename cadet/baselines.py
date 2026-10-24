"""Baseline and reference detectors used in the paper experiments."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from cadet.detectors import Alarm, StepResult
from cadet.residualization import LinearPURResidualizer, Residualizer
from cadet.utils import (
    as_vector,
    robust_scale,
    robust_whittle_ar1_fit,
    safe_alpha,
    anytime_alpha as anytime_alpha_fn,
)


@dataclass
class BaselineConfig:
    alpha: float = 0.05
    window: int = 80
    warmup: int = 160
    anytime: bool = True


class _WindowedDetector:
    name = "base"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.t = 0
        self.n_channels: int | None = None

    def _alpha_t(self) -> float:
        if self.config.anytime:
            return anytime_alpha_fn(self.config.alpha, self.t)
        return safe_alpha(self.config.alpha)

    def _alpha_per_channel(self) -> float:
        return safe_alpha(self._alpha_t() / max(int(self.n_channels or 1), 1))

    def _ensure_channels(self, d: int) -> None:
        if self.n_channels is None:
            self.n_channels = d
        elif self.n_channels != d:
            raise ValueError(f"expected {self.n_channels} channels, got {d}")

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        raise NotImplementedError


class MomentMonitor(_WindowedDetector):
    """Warm-up calibrated mean/scale monitor."""

    name = "Moment"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.warmup_values: list[list[float]] = []
        self.centers: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.windows: list[deque[float]] = []

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.warmup_values = [[] for _ in range(d)]
            self.windows = [deque(maxlen=self.config.window) for _ in range(d)]

    def _calibrate(self) -> None:
        arr = np.asarray(self.warmup_values, dtype=float)
        self.centers = np.median(arr, axis=1)
        self.scales = np.asarray([robust_scale(row) for row in arr], dtype=float)

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []

        if self.centers is None:
            for channel, value in enumerate(z):
                self.warmup_values[channel].append(float(value))
            if self.t >= self.config.warmup:
                self._calibrate()
            return StepResult(self.t, z.copy(), alarms=alarms)

        alpha_per = self._alpha_per_channel()
        threshold = norm.ppf(1.0 - alpha_per / 4.0)
        for channel, value in enumerate(z):
            window = self.windows[channel]
            window.append(float(value))
            if len(window) < self.config.window:
                continue
            arr = np.asarray(window, dtype=float)
            mean_z = abs(np.mean(arr) - self.centers[channel]) / (
                self.scales[channel] / math.sqrt(self.config.window)
            )
            scale_z = abs(math.log(robust_scale(arr) / self.scales[channel])) * math.sqrt(
                self.config.window / 2.0
            )
            statistic = max(float(mean_z), float(scale_z))
            p_value = min(1.0, 4.0 * norm.sf(statistic))
            candidate = Alarm(
                time=self.t,
                channel=channel,
                arm="moment",
                statistic=statistic,
                threshold=float(threshold),
                p_value=p_value,
                metadata={"mean_z": float(mean_z), "scale_z": float(scale_z)},
            )
            statistics.append(candidate)
            if statistic > threshold:
                alarms.append(candidate)
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


class CUSUMDetector(_WindowedDetector):
    """Two-sided Gaussian CUSUM baseline on warm-up standardized channels."""

    name = "CUSUM"

    def __init__(self, config: BaselineConfig | None = None, k: float = 0.5) -> None:
        super().__init__(config)
        self.k = float(k)
        self.warmup_values: list[list[float]] = []
        self.centers: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.g_pos: np.ndarray | None = None
        self.g_neg: np.ndarray | None = None

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.warmup_values = [[] for _ in range(d)]
            self.g_pos = np.zeros(d, dtype=float)
            self.g_neg = np.zeros(d, dtype=float)

    def _calibrate(self) -> None:
        arr = np.asarray(self.warmup_values, dtype=float)
        self.centers = np.median(arr, axis=1)
        self.scales = np.asarray([robust_scale(row) for row in arr], dtype=float)

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []

        if self.centers is None:
            for channel, value in enumerate(z):
                self.warmup_values[channel].append(float(value))
            if self.t >= self.config.warmup:
                self._calibrate()
            return StepResult(self.t, z.copy(), alarms=alarms)

        assert self.g_pos is not None and self.g_neg is not None
        alpha_per = self._alpha_per_channel()
        threshold = max(1.0, math.log(2.0 / alpha_per))
        standardized = (z - self.centers) / self.scales
        for channel, x in enumerate(standardized):
            self.g_pos[channel] = max(0.0, self.g_pos[channel] + float(x) - self.k)
            self.g_neg[channel] = max(0.0, self.g_neg[channel] - float(x) - self.k)
            statistic = max(self.g_pos[channel], self.g_neg[channel])
            p_value = min(1.0, math.exp(-statistic))
            candidate = Alarm(
                time=self.t,
                channel=channel,
                arm="cusum",
                statistic=float(statistic),
                threshold=float(threshold),
                p_value=p_value,
            )
            statistics.append(candidate)
            if statistic > threshold:
                alarms.append(candidate)
                self.g_pos[channel] = 0.0
                self.g_neg[channel] = 0.0
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


class ADWINDetector(_WindowedDetector):
    """ADWIN-style adaptive window baseline with Hoeffding cuts."""

    name = "ADWIN"

    def __init__(
        self,
        config: BaselineConfig | None = None,
        max_window: int = 400,
        min_segment: int = 20,
        scan_stride: int = 5,
    ) -> None:
        super().__init__(config)
        self.max_window = int(max_window)
        self.min_segment = int(min_segment)
        self.scan_stride = int(scan_stride)
        self.warmup_values: list[list[float]] = []
        self.centers: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.windows: list[deque[float]] = []

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.warmup_values = [[] for _ in range(d)]
            self.windows = [deque(maxlen=self.max_window) for _ in range(d)]

    def _calibrate(self) -> None:
        arr = np.asarray(self.warmup_values, dtype=float)
        self.centers = np.median(arr, axis=1)
        self.scales = np.asarray([robust_scale(row) for row in arr], dtype=float)

    def _squash(self, channel: int, value: float) -> float:
        assert self.centers is not None and self.scales is not None
        x = (value - self.centers[channel]) / (3.0 * self.scales[channel])
        return float(0.5 + 0.5 * math.tanh(x))

    def _scan(self, arr: np.ndarray, alpha: float) -> tuple[bool, float, float, int]:
        n = arr.size
        if n < 2 * self.min_segment:
            return False, 0.0, float("inf"), -1
        best_stat = 0.0
        best_threshold = float("inf")
        best_split = -1
        for split in range(self.min_segment, n - self.min_segment + 1, self.scan_stride):
            left = arr[:split]
            right = arr[split:]
            stat = abs(float(np.mean(left) - np.mean(right)))
            harmonic = 1.0 / (1.0 / left.size + 1.0 / right.size)
            threshold = math.sqrt(math.log(4.0 / safe_alpha(alpha)) / (2.0 * harmonic))
            if stat - threshold > best_stat - best_threshold:
                best_stat = stat
                best_threshold = threshold
                best_split = split
        return best_stat > best_threshold, best_stat, best_threshold, best_split

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []

        if self.centers is None:
            for channel, value in enumerate(z):
                self.warmup_values[channel].append(float(value))
            if self.t >= self.config.warmup:
                self._calibrate()
            return StepResult(self.t, z.copy(), alarms=alarms)

        alpha_per = self._alpha_per_channel()
        for channel, value in enumerate(z):
            window = self.windows[channel]
            window.append(self._squash(channel, float(value)))
            arr = np.asarray(window, dtype=float)
            fired, statistic, threshold, split = self._scan(arr, alpha_per)
            if split >= 0:
                candidate = Alarm(
                    time=self.t,
                    channel=channel,
                    arm="adwin",
                    statistic=statistic,
                    threshold=threshold,
                    p_value=None,
                    metadata={"split": split, "window_size": arr.size},
                )
                statistics.append(candidate)
            else:
                candidate = None
            if fired:
                assert candidate is not None
                alarms.append(candidate)
                kept = list(arr[split:])
                window.clear()
                window.extend(kept)
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


class PageHinkleyDetector(_WindowedDetector):
    """Two-sided Page-Hinkley mean-shift detector on warm-up standardized channels."""

    name = "PageHinkley"

    def __init__(self, config: BaselineConfig | None = None, delta: float = 0.05) -> None:
        super().__init__(config)
        self.delta = float(delta)
        self.warmup_values: list[list[float]] = []
        self.centers: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.g_pos: np.ndarray | None = None
        self.g_neg: np.ndarray | None = None

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.warmup_values = [[] for _ in range(d)]
            self.g_pos = np.zeros(d, dtype=float)
            self.g_neg = np.zeros(d, dtype=float)

    def _calibrate(self) -> None:
        arr = np.asarray(self.warmup_values, dtype=float)
        self.centers = np.median(arr, axis=1)
        self.scales = np.asarray([robust_scale(row) for row in arr], dtype=float)

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []

        if self.centers is None:
            for channel, value in enumerate(z):
                self.warmup_values[channel].append(float(value))
            if self.t >= self.config.warmup:
                self._calibrate()
            return StepResult(self.t, z.copy(), alarms=alarms)

        assert self.g_pos is not None and self.g_neg is not None
        alpha_per = self._alpha_per_channel()
        threshold = max(2.0, math.log(2.0 / alpha_per))
        standardized = (z - self.centers) / self.scales
        for channel, value in enumerate(standardized):
            x = float(value)
            self.g_pos[channel] = max(0.0, self.g_pos[channel] + x - self.delta)
            self.g_neg[channel] = max(0.0, self.g_neg[channel] - x - self.delta)
            statistic = max(self.g_pos[channel], self.g_neg[channel])
            p_value = min(1.0, 2.0 * math.exp(-statistic))
            candidate = Alarm(
                time=self.t,
                channel=channel,
                arm="page_hinkley",
                statistic=float(statistic),
                threshold=float(threshold),
                p_value=p_value,
            )
            statistics.append(candidate)
            if statistic > threshold:
                alarms.append(candidate)
                self.g_pos[channel] = 0.0
                self.g_neg[channel] = 0.0
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


def _median_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 2:
        return 1.0
    distances = np.abs(values[:, None] - values[None, :])
    upper = distances[np.triu_indices(values.size, k=1)]
    median = float(np.median(upper[upper > 0.0])) if np.any(upper > 0.0) else 1.0
    return max(median, 1e-6)


def _rbf_mmd2(reference: np.ndarray, detection: np.ndarray) -> tuple[float, float]:
    x = np.asarray(reference, dtype=float).reshape(-1)
    y = np.asarray(detection, dtype=float).reshape(-1)
    bandwidth = _median_bandwidth(np.concatenate([x, y]))
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    k_xx = np.exp(-gamma * (x[:, None] - x[None, :]) ** 2)
    k_yy = np.exp(-gamma * (y[:, None] - y[None, :]) ** 2)
    k_xy = np.exp(-gamma * (x[:, None] - y[None, :]) ** 2)
    if x.size > 1:
        xx = (np.sum(k_xx) - np.trace(k_xx)) / (x.size * (x.size - 1))
    else:
        xx = 0.0
    if y.size > 1:
        yy = (np.sum(k_yy) - np.trace(k_yy)) / (y.size * (y.size - 1))
    else:
        yy = 0.0
    statistic = float(max(0.0, xx + yy - 2.0 * np.mean(k_xy)))
    return statistic, bandwidth


class KernelMMDDetector(_WindowedDetector):
    """Fixed-reference RBF-MMD two-sample detector."""

    name = "MMD"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.warmup_values: list[list[float]] = []
        self.reference: list[np.ndarray] | None = None
        self.windows: list[deque[float]] = []

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.warmup_values = [[] for _ in range(d)]
            self.windows = [deque(maxlen=self.config.window) for _ in range(d)]

    def _calibrate(self) -> None:
        self.reference = []
        for values in self.warmup_values:
            arr = np.asarray(values[-self.config.window :], dtype=float)
            self.reference.append(arr.copy())

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []

        if self.reference is None:
            for channel, value in enumerate(z):
                self.warmup_values[channel].append(float(value))
            if self.t >= self.config.warmup:
                self._calibrate()
            return StepResult(self.t, z.copy(), alarms=alarms)

        alpha_per = self._alpha_per_channel()
        for channel, value in enumerate(z):
            window = self.windows[channel]
            window.append(float(value))
            if len(window) < self.config.window:
                continue
            detection = np.asarray(window, dtype=float)
            statistic, bandwidth = _rbf_mmd2(self.reference[channel], detection)
            n_eff = min(len(self.reference[channel]), detection.size)
            threshold = 2.0 * math.sqrt(math.log(2.0 / alpha_per) / max(n_eff, 1))
            p_value = min(1.0, 2.0 * math.exp(-max(n_eff, 1) * statistic * statistic / 4.0))
            candidate = Alarm(
                time=self.t,
                channel=channel,
                arm="mmd",
                statistic=statistic,
                threshold=threshold,
                p_value=p_value,
                metadata={"bandwidth": bandwidth},
            )
            statistics.append(candidate)
            if statistic > threshold:
                alarms.append(candidate)
                window.clear()
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


def _energy_distance(reference: np.ndarray, detection: np.ndarray) -> float:
    x = np.asarray(reference, dtype=float).reshape(-1)
    y = np.asarray(detection, dtype=float).reshape(-1)
    xy = np.mean(np.abs(x[:, None] - y[None, :]))
    xx = np.mean(np.abs(x[:, None] - x[None, :]))
    yy = np.mean(np.abs(y[:, None] - y[None, :]))
    return float(max(0.0, 2.0 * xy - xx - yy))


class EnergyDistanceDetector(_WindowedDetector):
    """Fixed-reference energy-distance two-sample detector."""

    name = "Energy"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.warmup_values: list[list[float]] = []
        self.reference: list[np.ndarray] | None = None
        self.reference_scales: np.ndarray | None = None
        self.windows: list[deque[float]] = []

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.warmup_values = [[] for _ in range(d)]
            self.windows = [deque(maxlen=self.config.window) for _ in range(d)]

    def _calibrate(self) -> None:
        reference: list[np.ndarray] = []
        scales: list[float] = []
        for values in self.warmup_values:
            arr = np.asarray(values[-self.config.window :], dtype=float)
            reference.append(arr.copy())
            scales.append(robust_scale(arr))
        self.reference = reference
        self.reference_scales = np.asarray(scales, dtype=float)

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []

        if self.reference is None:
            for channel, value in enumerate(z):
                self.warmup_values[channel].append(float(value))
            if self.t >= self.config.warmup:
                self._calibrate()
            return StepResult(self.t, z.copy(), alarms=alarms)

        assert self.reference_scales is not None
        alpha_per = self._alpha_per_channel()
        for channel, value in enumerate(z):
            window = self.windows[channel]
            window.append(float(value))
            if len(window) < self.config.window:
                continue
            detection = np.asarray(window, dtype=float)
            raw_statistic = _energy_distance(self.reference[channel], detection)
            statistic = raw_statistic / max(self.reference_scales[channel], 1e-12)
            n_eff = min(len(self.reference[channel]), detection.size)
            threshold = 4.0 * math.sqrt(math.log(2.0 / alpha_per) / max(n_eff, 1))
            p_value = min(1.0, 2.0 * math.exp(-max(n_eff, 1) * statistic * statistic / 16.0))
            candidate = Alarm(
                time=self.t,
                channel=channel,
                arm="energy",
                statistic=statistic,
                threshold=threshold,
                p_value=p_value,
                metadata={"raw_energy": raw_statistic},
            )
            statistics.append(candidate)
            if statistic > threshold:
                alarms.append(candidate)
                window.clear()
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


class BOCPDGaussianDetector(_WindowedDetector):
    """Gaussian predictive-surprise baseline in the style of BOCPD."""

    name = "BOCPD"

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.warmup_values: list[list[float]] = []
        self.centers: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.windows: list[deque[float]] = []

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.warmup_values = [[] for _ in range(d)]
            self.windows = [deque(maxlen=self.config.window) for _ in range(d)]

    def _calibrate(self) -> None:
        arr = np.asarray(self.warmup_values, dtype=float)
        self.centers = np.median(arr, axis=1)
        self.scales = np.asarray([robust_scale(row) for row in arr], dtype=float)

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []

        if self.centers is None:
            for channel, value in enumerate(z):
                self.warmup_values[channel].append(float(value))
            if self.t >= self.config.warmup:
                self._calibrate()
            return StepResult(self.t, z.copy(), alarms=alarms)

        assert self.scales is not None
        alpha_per = self._alpha_per_channel()
        threshold = float(chi2.ppf(1.0 - alpha_per, df=self.config.window) / self.config.window)
        standardized = (z - self.centers) / self.scales
        for channel, value in enumerate(standardized):
            window = self.windows[channel]
            window.append(float(value) ** 2)
            if len(window) < self.config.window:
                continue
            statistic = float(np.mean(window))
            p_value = float(chi2.sf(statistic * self.config.window, df=self.config.window))
            candidate = Alarm(
                time=self.t,
                channel=channel,
                arm="bocpd_gaussian",
                statistic=statistic,
                threshold=threshold,
                p_value=p_value,
                metadata={
                    "predictive_center": float(self.centers[channel]),
                    "predictive_scale": float(self.scales[channel]),
                },
            )
            statistics.append(candidate)
            if statistic > threshold:
                alarms.append(candidate)
                window.clear()
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


class RobustScaleGLRDetector(_WindowedDetector):
    """Robust window-limited scale GLR reference."""

    name = "ScaleGLR"

    def __init__(
        self,
        config: BaselineConfig | None = None,
        min_segment: int = 20,
        scan_stride: int = 5,
    ) -> None:
        super().__init__(config)
        self.min_segment = int(min_segment)
        self.scan_stride = int(scan_stride)
        self.windows: list[deque[float]] = []

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.windows = [deque(maxlen=self.config.window) for _ in range(d)]

    def _scan(self, arr: np.ndarray, alpha: float) -> tuple[bool, float, float, int, float]:
        n = arr.size
        if n < 2 * self.min_segment:
            return False, 0.0, float("inf"), -1, 1.0
        splits = list(range(self.min_segment, n - self.min_segment + 1, self.scan_stride))
        best = 0.0
        best_split = -1
        for split in splits:
            left = arr[:split]
            right = arr[split:]
            ratio = robust_scale(right) / robust_scale(left)
            stat = (left.size * right.size / n) * math.log(max(ratio, 1.0 / ratio)) ** 2
            if stat > best:
                best = float(stat)
                best_split = split
        split_alpha = safe_alpha(alpha / max(len(splits), 1))
        threshold = float(chi2.ppf(1.0 - split_alpha, df=1))
        p_value = min(1.0, len(splits) * float(chi2.sf(best, df=1)))
        return best > threshold, best, threshold, best_split, p_value

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []
        alpha_per = self._alpha_per_channel()
        suppress = self.t <= self.config.warmup
        for channel, value in enumerate(z):
            window = self.windows[channel]
            window.append(float(value))
            arr = np.asarray(window, dtype=float)
            fired, statistic, threshold, split, p_value = self._scan(arr, alpha_per)
            if split >= 0:
                candidate = Alarm(
                    time=self.t,
                    channel=channel,
                    arm="scale_glr",
                    statistic=statistic,
                    threshold=threshold,
                    p_value=p_value,
                    metadata={"split": split},
                )
                statistics.append(candidate)
            else:
                candidate = None
            if fired and not suppress:
                assert candidate is not None
                alarms.append(candidate)
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


class RobustDynamicsGLRDetector(_WindowedDetector):
    """Robust window-limited dynamics GLR reference using Whittle AR(1) fits."""

    name = "DynamicsGLR"

    def __init__(
        self,
        config: BaselineConfig | None = None,
        min_segment: int = 25,
        scan_stride: int = 5,
    ) -> None:
        super().__init__(config)
        self.min_segment = int(min_segment)
        self.scan_stride = int(scan_stride)
        self.prev: np.ndarray | None = None
        self.increments: list[deque[float]] = []

    def _ensure_channels(self, d: int) -> None:
        old = self.n_channels
        super()._ensure_channels(d)
        if old is None:
            self.increments = [deque(maxlen=self.config.window) for _ in range(d)]

    def _scan(self, arr: np.ndarray, alpha: float) -> tuple[bool, float, float, int, float]:
        n = arr.size
        if n < 2 * self.min_segment:
            return False, 0.0, float("inf"), -1, 1.0
        splits = list(range(self.min_segment, n - self.min_segment + 1, self.scan_stride))
        best = 0.0
        best_split = -1
        _, _, nll_pool = robust_whittle_ar1_fit(arr)
        for split in splits:
            left = arr[:split]
            right = arr[split:]
            _, _, nll_left = robust_whittle_ar1_fit(left)
            _, _, nll_right = robust_whittle_ar1_fit(right)
            stat = max(0.0, 2.0 * float(nll_pool - nll_left - nll_right))
            if stat > best:
                best = stat
                best_split = split
        split_alpha = safe_alpha(alpha / max(len(splits), 1))
        threshold = float(chi2.ppf(1.0 - split_alpha, df=1))
        p_value = min(1.0, len(splits) * float(chi2.sf(best, df=1)))
        return best > threshold, best, threshold, best_split, p_value

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []
        if self.prev is None:
            self.prev = z.copy()
            return StepResult(self.t, z.copy(), alarms=alarms)

        inc = z - self.prev
        self.prev = z.copy()
        alpha_per = self._alpha_per_channel()
        suppress = self.t <= self.config.warmup
        for channel, value in enumerate(inc):
            window = self.increments[channel]
            window.append(float(value))
            arr = np.asarray(window, dtype=float)
            fired, statistic, threshold, split, p_value = self._scan(arr, alpha_per)
            if split >= 0:
                candidate = Alarm(
                    time=self.t,
                    channel=channel,
                    arm="dynamics_glr",
                    statistic=statistic,
                    threshold=threshold,
                    p_value=p_value,
                    metadata={"split": split},
                )
                statistics.append(candidate)
            else:
                candidate = None
            if fired and not suppress:
                assert candidate is not None
                alarms.append(candidate)
        return StepResult(self.t, z.copy(), alarms=alarms, statistics=statistics)


class ResidualizedDetector:
    """Run any baseline detector on a residualized copy of the monitored stream.

    Reviewers asked whether the online baselines see the raw stream ``z_t`` or the
    PUR-residualized stream ``r_t``. In the default factories they see ``z_t``: the
    baselines ignore ``endogenous_delta`` entirely. This wrapper feeds a baseline the
    same residual stream CADET consumes, so a ``PUR + baseline-statistic`` ablation can
    isolate how much of any CADET advantage comes from residualization rather than from
    the scale/dynamics statistics themselves.

    The wrapper is deliberately transparent: it forwards ``name``, ``config`` and
    ``n_channels`` so that the calibration and evaluation code paths treat it exactly
    like the detector it wraps.
    """

    def __init__(
        self,
        inner: Any,
        residualizer: Residualizer | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.inner = inner
        self.residualizer = residualizer if residualizer is not None else LinearPURResidualizer()
        self.residualizer.reset()
        self.name = name or f"{getattr(inner, 'name', 'baseline')}_PUR"

    @property
    def config(self) -> Any:
        return self.inner.config

    @property
    def t(self) -> int:
        return self.inner.t

    @property
    def n_channels(self) -> int | None:
        return self.inner.n_channels

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        residual = self.residualizer.update(z_t, endogenous_delta)
        # The inner detector must not subtract the increment a second time.
        return self.inner.update(residual, endogenous_delta=None)
