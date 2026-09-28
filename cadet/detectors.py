"""CADET detector and its scale/dynamics arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cadet.residualization import LinearPURResidualizer, Residualizer
from cadet.utils import (
    as_vector,
    cauchy_combination_p_value,
    dkw_scale_p_value,
    dkw_scale_threshold,
    dynamics_p_value,
    dynamics_radius,
    estimate_lambda_phi,
    multiplicity_corrected_z,
    P2IQR,
    P2Quantile,
    robust_scale,
    safe_alpha,
    anytime_alpha as anytime_alpha_fn,
)


@dataclass
class Alarm:
    time: int
    channel: int | None
    arm: str
    statistic: float
    threshold: float
    p_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmResult:
    ready: bool
    fired: bool = False
    statistic: float = 0.0
    threshold: float = float("inf")
    p_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    time: int
    residual: np.ndarray
    scale: list[ArmResult] = field(default_factory=list)
    dynamics: list[ArmResult] = field(default_factory=list)
    location: list[ArmResult] = field(default_factory=list)
    alarms: list[Alarm] = field(default_factory=list)
    statistics: list[Alarm] = field(default_factory=list)

    @property
    def alarm(self) -> Alarm | None:
        return self.alarms[0] if self.alarms else None


@dataclass
class CADETConfig:
    alpha: float = 0.05
    scale_window: int = 80
    dynamics_window: int = 80
    warmup: int = 160
    lambda_phi: float = 1.0
    grid_size: int = 128
    grid_max: float = 10.0
    anytime: bool = True
    arms: tuple[str, ...] = ("scale", "dynamics")
    dense_cauchy: bool = False
    dense_cauchy_fraction: float = 0.5
    reset_after_alarm: bool = False
    center_mode: str = "reference"
    lambda_phi_mode: str = "fixed"
    # Legacy autocorrelation diagnostic. Its status reports only whether the
    # sample summary exceeds the configured multiplier, not whether a certified
    # mixing-matrix bound holds. Zero disables the diagnostic. The benchmark
    # retains these historical score formulas and uses empirical calibration.
    lambda_phi_monitor_block: int = 0
    reference_center_warmup: int | None = None

    def __post_init__(self) -> None:
        valid = {"scale", "dynamics", "location", "anchored_location"}
        unknown = set(self.arms) - valid
        if unknown:
            raise ValueError(f"unknown CADET arm(s): {sorted(unknown)}")
        if self.scale_window < 2 or self.dynamics_window < 3:
            raise ValueError("scale_window must be >=2 and dynamics_window must be >=3")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if self.grid_max <= 0.0:
            raise ValueError("grid_max must be positive")
        if not 0.0 < self.dense_cauchy_fraction < 1.0:
            raise ValueError("dense_cauchy_fraction must lie in (0, 1)")
        if self.center_mode not in {"reference", "block"}:
            raise ValueError("center_mode must be 'reference' or 'block'")
        if self.lambda_phi_mode not in {"fixed", "warmup"}:
            raise ValueError("lambda_phi_mode must be 'fixed' or 'warmup'")
        if self.reference_center_warmup is not None and self.reference_center_warmup < 5:
            raise ValueError("reference_center_warmup must be at least 5 when provided")


class ScaleArm:
    """Fixed-grid streaming dispersion-ECDF detector for one channel."""

    def __init__(
        self,
        window: int,
        lambda_phi: float = 1.0,
        grid_size: int = 128,
        grid_max: float = 10.0,
        center_mode: str = "reference",
        reference_center_warmup: int | None = None,
    ) -> None:
        self.window = int(window)
        self.lambda_phi = float(lambda_phi)
        if center_mode not in {"reference", "block"}:
            raise ValueError("center_mode must be 'reference' or 'block'")
        self.center_mode = center_mode
        self.reference_center_warmup = reference_center_warmup
        self.thresholds = np.linspace(0.0, float(grid_max), int(grid_size))
        self.reference_counts: np.ndarray | None = None
        self.reference_n = 0
        self.reference_iqr = 0.0
        self.reference_center: float | None = None
        self._reset_detection_block()

    def reset(self) -> None:
        self.reference_counts = None
        self.reference_n = 0
        self.reference_iqr = 0.0
        self.reference_center = None
        self._reset_detection_block()

    def _reset_detection_block(self) -> None:
        self.detection_counts = np.zeros(self.thresholds.size + 1, dtype=int)
        self.detection_n = 0
        self.detection_p2 = P2IQR()
        self.detection_values: list[float] = []

    def _add_to_detection(self, value: float) -> None:
        value = float(value)
        self.detection_values.append(value)
        self.detection_p2.update(value)
        if self.center_mode == "reference" and self.reference_center is not None:
            center = self.reference_center
        else:
            center = self.detection_p2.median
        dispersion = abs(value - center)
        bin_index = int(np.searchsorted(self.thresholds, dispersion, side="left"))
        bin_index = min(bin_index, self.thresholds.size)
        self.detection_counts[bin_index] += 1
        self.detection_n += 1

    def _counts_for_center(self, values: list[float], center: float) -> np.ndarray:
        counts = np.zeros(self.thresholds.size + 1, dtype=int)
        for value in values:
            dispersion = abs(float(value) - center)
            bin_index = int(np.searchsorted(self.thresholds, dispersion, side="left"))
            counts[min(bin_index, self.thresholds.size)] += 1
        return counts

    def _ecdf(self, counts: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(self.thresholds.size, dtype=float)
        return np.cumsum(counts[:-1]) / float(n)

    def update(self, value: float, alpha: float) -> ArmResult:
        self._add_to_detection(float(value))
        if self.detection_n < self.window:
            return ArmResult(ready=False)

        det_n = self.detection_n
        det_iqr = self.detection_p2.iqr
        det_center = self.detection_p2.median
        if self.reference_counts is None:
            center_values = self.detection_values
            if self.reference_center_warmup is not None:
                center_values = center_values[: self.reference_center_warmup]
            det_center = float(np.median(np.asarray(center_values, dtype=float)))
            det_counts = self._counts_for_center(self.detection_values, det_center)
            self.reference_counts = det_counts
            self.reference_n = det_n
            self.reference_iqr = det_iqr
            self.reference_center = det_center
            self._reset_detection_block()
            return ArmResult(
                ready=False,
                metadata={
                    "reference_initialized": True,
                    "center_mode": self.center_mode,
                    "reference_center": self.reference_center,
                },
            )

        det_counts = self.detection_counts.copy()

        reference_cdf = self._ecdf(self.reference_counts, self.reference_n)
        detection_cdf = self._ecdf(det_counts, det_n)
        statistic = float(np.max(np.abs(detection_cdf - reference_cdf)))
        threshold = dkw_scale_threshold(alpha, self.window, self.lambda_phi)
        p_value = dkw_scale_p_value(statistic, self.window, self.lambda_phi)
        result = ArmResult(
            ready=True,
            fired=statistic > threshold,
            statistic=statistic,
            threshold=threshold,
            p_value=p_value,
            metadata={
                "iqr_reference": self.reference_iqr,
                "iqr_detection": det_iqr,
                "iqr_ratio": det_iqr / max(self.reference_iqr, 1e-12),
                "lambda_phi": self.lambda_phi,
                "grid_size": self.thresholds.size,
                "center_mode": self.center_mode,
                "reference_center": self.reference_center,
            },
        )
        self.reference_counts = det_counts
        self.reference_n = det_n
        self.reference_iqr = det_iqr
        if self.center_mode == "block":
            self.reference_center = det_center
        self._reset_detection_block()
        return result


class DynamicsArm:
    """Streaming median-sign level and magnitude flip-rate detector."""

    def __init__(self, window: int, pilot_size: int, lambda_phi: float = 1.0) -> None:
        self.window = int(window)
        self.pilot_size = max(int(pilot_size), self.window + 2)
        self.lambda_phi = float(lambda_phi)
        self.prev: float | None = None
        self.increment_median = P2Quantile(0.5)
        self.abs_increment_median = P2Quantile(0.5)
        self.prev_level_sign: bool | None = None
        self.prev_magnitude_sign: bool | None = None
        self.pilot_level_flips = 0
        self.pilot_magnitude_flips = 0
        self.pilot_count = 0
        self.block_level_flips = 0
        self.block_magnitude_flips = 0
        self.block_count = 0
        self.p0: float | None = None
        self.q0: float | None = None

    def reset(self) -> None:
        self.prev = None
        self.increment_median = P2Quantile(0.5)
        self.abs_increment_median = P2Quantile(0.5)
        self.prev_level_sign = None
        self.prev_magnitude_sign = None
        self.pilot_level_flips = 0
        self.pilot_magnitude_flips = 0
        self.pilot_count = 0
        self.block_level_flips = 0
        self.block_magnitude_flips = 0
        self.block_count = 0
        self.p0 = None
        self.q0 = None

    @property
    def calibrated(self) -> bool:
        return self.p0 is not None and self.q0 is not None

    def _calibrate(self) -> None:
        denom = max(self.pilot_count, 1)
        self.p0 = self.pilot_level_flips / denom
        self.q0 = self.pilot_magnitude_flips / denom

    def _reset_block(self) -> None:
        self.block_level_flips = 0
        self.block_magnitude_flips = 0
        self.block_count = 0

    def update(self, value: float, alpha: float) -> ArmResult:
        value = float(value)
        if self.prev is None:
            self.prev = value
            return ArmResult(ready=False)

        inc = value - self.prev
        self.prev = value

        inc_median = self.increment_median.update(float(inc))
        abs_median = self.abs_increment_median.update(abs(float(inc)))
        level_sign = bool(inc >= inc_median)
        magnitude_sign = bool(abs(inc) >= abs_median)
        level_flip = self.prev_level_sign is not None and level_sign != self.prev_level_sign
        magnitude_flip = (
            self.prev_magnitude_sign is not None and magnitude_sign != self.prev_magnitude_sign
        )
        self.prev_level_sign = level_sign
        self.prev_magnitude_sign = magnitude_sign

        if not self.calibrated:
            if self.pilot_count > 0:
                self.pilot_level_flips += int(level_flip)
                self.pilot_magnitude_flips += int(magnitude_flip)
            self.pilot_count += 1
            if self.pilot_count >= self.pilot_size:
                self._calibrate()
            return ArmResult(ready=False)

        self.block_level_flips += int(level_flip)
        self.block_magnitude_flips += int(magnitude_flip)
        self.block_count += 1
        if self.block_count < self.window:
            return ArmResult(ready=False)

        level = self.block_level_flips / max(self.block_count, 1)
        magnitude = self.block_magnitude_flips / max(self.block_count, 1)
        rho = dynamics_radius(alpha, self.window, self.lambda_phi)
        level_dev = abs(level - float(self.p0))
        magnitude_dev = abs(magnitude - float(self.q0))
        max_dev = max(level_dev, magnitude_dev)
        statistic = max_dev / max(rho, 1e-12)
        p_value = dynamics_p_value(max_dev, self.window, self.lambda_phi)
        result = ArmResult(
            ready=True,
            fired=statistic > 1.0,
            statistic=statistic,
            threshold=1.0,
            p_value=p_value,
            metadata={
                "level_flip": level,
                "magnitude_flip": magnitude,
                "p0": self.p0,
                "q0": self.q0,
                "rho": rho,
            },
        )
        self._reset_block()
        return result


class LocationArm:
    """Blockwise robust location-shift detector for one channel.

    This arm is intended for empirical-threshold calibration on deployed streams
    whose main change is a reward or task-signal level shift. Its nominal
    threshold is conservative and mainly keeps the non-empirical API usable.
    """

    def __init__(
        self,
        window: int,
        lambda_phi: float = 1.0,
        scale_floor: float = 1e-3,
        trim_fraction: float = 0.10,
    ) -> None:
        self.window = int(window)
        self.lambda_phi = float(lambda_phi)
        self.scale_floor = float(scale_floor)
        self.trim_fraction = float(trim_fraction)
        self.reference_values: np.ndarray | None = None
        self.reference_location = 0.0
        self.reference_scale = 1.0
        self.detection_values: list[float] = []

    def reset(self) -> None:
        self.reference_values = None
        self.reference_location = 0.0
        self.reference_scale = 1.0
        self.detection_values = []

    def _block_scale(self, values: np.ndarray) -> float:
        return max(robust_scale(values), self.scale_floor)

    def _block_location(self, values: np.ndarray) -> float:
        arr = np.sort(np.asarray(values, dtype=float).reshape(-1))
        if arr.size == 0:
            return 0.0
        trim = int(np.floor(self.trim_fraction * arr.size))
        if trim > 0 and 2 * trim < arr.size:
            arr = arr[trim : arr.size - trim]
        return float(np.mean(arr))

    def update(self, value: float, alpha: float) -> ArmResult:
        self.detection_values.append(float(value))
        if len(self.detection_values) < self.window:
            return ArmResult(ready=False)

        block = np.asarray(self.detection_values, dtype=float)
        if self.reference_values is None:
            self.reference_values = block
            self.reference_location = self._block_location(block)
            self.reference_scale = self._block_scale(block)
            self.detection_values = []
            return ArmResult(
                ready=False,
                metadata={
                    "reference_initialized": True,
                    "reference_location": self.reference_location,
                    "reference_scale": self.reference_scale,
                },
            )

        assert self.reference_values is not None
        block_location = self._block_location(block)
        block_scale = self._block_scale(block)
        pooled_scale = self._block_scale(np.concatenate([self.reference_values, block]))
        scale = max(float(self.reference_scale), block_scale, pooled_scale, 1e-12)
        statistic = abs(block_location - self.reference_location) / scale
        n_eff = max(float(self.window) / max(self.lambda_phi**2, 1.0), 1.0)
        threshold = float(np.sqrt(2.0 * np.log(2.0 / safe_alpha(alpha)) / n_eff))
        p_value = float(min(1.0, 2.0 * np.exp(-0.5 * n_eff * statistic**2)))
        result = ArmResult(
            ready=True,
            fired=statistic > threshold,
            statistic=statistic,
            threshold=threshold,
            p_value=p_value,
            metadata={
                "reference_location": self.reference_location,
                "block_location": block_location,
                "reference_scale": scale,
                "reference_block_scale": self.reference_scale,
                "detection_block_scale": block_scale,
                "pooled_block_scale": pooled_scale,
                "trim_fraction": self.trim_fraction,
                "lambda_phi": self.lambda_phi,
            },
        )
        self.reference_values = block
        self.reference_location = block_location
        self.reference_scale = block_scale
        self.detection_values = []
        return result


class AnchoredLocationArm(LocationArm):
    """Robust location-shift detector with a fixed initial reference block."""

    def update(self, value: float, alpha: float) -> ArmResult:
        self.detection_values.append(float(value))
        if len(self.detection_values) < self.window:
            return ArmResult(ready=False)

        block = np.asarray(self.detection_values, dtype=float)
        if self.reference_values is None:
            self.reference_values = block
            self.reference_location = self._block_location(block)
            self.reference_scale = self._block_scale(block)
            self.detection_values = []
            return ArmResult(
                ready=False,
                metadata={
                    "reference_initialized": True,
                    "reference_location": self.reference_location,
                    "reference_scale": self.reference_scale,
                    "anchored": True,
                },
            )

        assert self.reference_values is not None
        block_location = self._block_location(block)
        block_scale = self._block_scale(block)
        pooled_scale = self._block_scale(np.concatenate([self.reference_values, block]))
        scale = max(float(self.reference_scale), block_scale, pooled_scale, 1e-12)
        statistic = abs(block_location - self.reference_location) / scale
        n_eff = max(float(self.window) / max(self.lambda_phi**2, 1.0), 1.0)
        threshold = float(np.sqrt(2.0 * np.log(2.0 / safe_alpha(alpha)) / n_eff))
        p_value = float(min(1.0, 2.0 * np.exp(-0.5 * n_eff * statistic**2)))
        result = ArmResult(
            ready=True,
            fired=statistic > threshold,
            statistic=statistic,
            threshold=threshold,
            p_value=p_value,
            metadata={
                "reference_location": self.reference_location,
                "block_location": block_location,
                "reference_scale": scale,
                "reference_block_scale": self.reference_scale,
                "detection_block_scale": block_scale,
                "pooled_block_scale": pooled_scale,
                "trim_fraction": self.trim_fraction,
                "lambda_phi": self.lambda_phi,
                "anchored": True,
            },
        )
        self.detection_values = []
        return result


class CADETDetector:
    """Streaming CADET detector with PUR, scale/dynamics arms, and late fusion."""

    def __init__(
        self,
        config: CADETConfig | None = None,
        residualizer: Residualizer | None = None,
    ) -> None:
        self.config = config or CADETConfig()
        self.residualizer = residualizer or LinearPURResidualizer()
        self.t = 0
        self.segment_t = 0
        self.n_channels: int | None = None
        self.scale_arms: list[ScaleArm] = []
        self.dynamics_arms: list[DynamicsArm] = []
        self.location_arms: list[LocationArm] = []
        self.anchored_location_arms: list[AnchoredLocationArm] = []
        self._alarm_count = 0
        self._warmup_residuals: list[np.ndarray] = []
        self._lambda_calibrated = False
        self._monitor_block: list[np.ndarray] = []
        self.lambda_phi_monitor: dict[str, float] = {
            "blocks": 0.0,
            "violations": 0.0,
            "max_estimate": 0.0,
            "deployed": 0.0,
        }

    def reset(self) -> None:
        self.t = 0
        self.segment_t = 0
        self.n_channels = None
        self.scale_arms = []
        self.dynamics_arms = []
        self.location_arms = []
        self.anchored_location_arms = []
        self._alarm_count = 0
        self._warmup_residuals = []
        self._lambda_calibrated = False
        self._monitor_block = []
        self.residualizer.reset()

    def _reset_segment(self) -> None:
        d = self.n_channels
        self.segment_t = 0
        self._warmup_residuals = []
        self._lambda_calibrated = False
        self._monitor_block = []
        if d is None:
            return
        self.scale_arms = [
            ScaleArm(
                self.config.scale_window,
                self.config.lambda_phi,
                self.config.grid_size,
                self.config.grid_max,
                self.config.center_mode,
                self.config.reference_center_warmup,
            )
            for _ in range(d)
        ]
        self.dynamics_arms = [
            DynamicsArm(self.config.dynamics_window, self.config.warmup, self.config.lambda_phi)
            for _ in range(d)
        ]
        self.location_arms = [
            LocationArm(self.config.scale_window, self.config.lambda_phi) for _ in range(d)
        ]
        self.anchored_location_arms = [
            AnchoredLocationArm(self.config.scale_window, self.config.lambda_phi)
            for _ in range(d)
        ]
        self.residualizer.reset()

    def _ensure_channels(self, d: int) -> None:
        if self.n_channels is None:
            self.n_channels = d
            self.scale_arms = [
                ScaleArm(
                    self.config.scale_window,
                    self.config.lambda_phi,
                    self.config.grid_size,
                    self.config.grid_max,
                    self.config.center_mode,
                    self.config.reference_center_warmup,
                )
                for _ in range(d)
            ]
            self.dynamics_arms = [
                DynamicsArm(self.config.dynamics_window, self.config.warmup, self.config.lambda_phi)
                for _ in range(d)
            ]
            self.location_arms = [
                LocationArm(self.config.scale_window, self.config.lambda_phi) for _ in range(d)
            ]
            self.anchored_location_arms = [
                AnchoredLocationArm(self.config.scale_window, self.config.lambda_phi)
                for _ in range(d)
            ]
            return
        if self.n_channels != d:
            raise ValueError(f"expected {self.n_channels} channels, got {d}")

    def _maybe_calibrate_lambda_phi(self, residual: np.ndarray) -> None:
        if self.config.lambda_phi_mode != "warmup" or self._lambda_calibrated:
            return
        if self.config.warmup <= 0:
            self._lambda_calibrated = True
            return
        self._warmup_residuals.append(residual.copy())
        if self.segment_t < self.config.warmup:
            return
        history = np.asarray(self._warmup_residuals, dtype=float)
        if history.ndim != 2 or history.shape[1] != len(self.scale_arms):
            self._lambda_calibrated = True
            return
        for channel in range(history.shape[1]):
            estimate = max(self.config.lambda_phi, estimate_lambda_phi(history[:, channel]))
            if "scale" in self.config.arms:
                self.scale_arms[channel].lambda_phi = estimate
            if "dynamics" in self.config.arms:
                self.dynamics_arms[channel].lambda_phi = estimate
            if "location" in self.config.arms:
                self.location_arms[channel].lambda_phi = estimate
            if "anchored_location" in self.config.arms:
                self.anchored_location_arms[channel].lambda_phi = estimate
        self._lambda_calibrated = True

    def _monitor_lambda_phi(self, residual: np.ndarray) -> None:
        """Re-estimate Lambda_phi on rolling post-warm-up blocks.

        The false-alarm theorems require the deployed Lambda_phi to be a genuine
        conservative upper bound on the stream's mixing coefficients, but nothing in
        the streaming loop checks that. This monitor recomputes the warm-up estimator
        on each later block and counts how often the running estimate exceeds the
        deployed constant, so an operator can see that the stated alpha level is no
        longer certified rather than discovering it from a burst of alarms.
        """

        block = int(self.config.lambda_phi_monitor_block)
        if block <= 0 or not self.scale_arms:
            return
        self._monitor_block.append(residual.copy())
        if len(self._monitor_block) < block:
            return
        history = np.asarray(self._monitor_block, dtype=float)
        self._monitor_block = []
        if history.ndim != 2:
            return
        deployed = max(
            (arm.lambda_phi for arm in self.scale_arms),
            default=float(self.config.lambda_phi),
        )
        # The estimator is applied once per lag and once per channel on every block, so a
        # per-lag two-sigma cut would flag almost every block of independent data. The
        # cut-off is corrected for the number of simultaneous tests actually performed.
        max_lag = min(50, max(history.shape[0] - 1, 1))
        floor_z = multiplicity_corrected_z(max_lag * history.shape[1])
        estimate = max(
            (
                estimate_lambda_phi(history[:, c], max_lag=max_lag, noise_floor_z=floor_z)
                for c in range(history.shape[1])
            ),
            default=1.0,
        )
        self.lambda_phi_monitor["blocks"] += 1.0
        self.lambda_phi_monitor["deployed"] = float(deployed)
        self.lambda_phi_monitor["max_estimate"] = max(
            self.lambda_phi_monitor["max_estimate"], float(estimate)
        )
        if estimate > deployed:
            self.lambda_phi_monitor["violations"] += 1.0

    @property
    def lambda_phi_status(self) -> str:
        """``"not_monitored"``, ``"conservative"``, or ``"calibration_invalid"``."""

        if self.lambda_phi_monitor["blocks"] <= 0.0:
            return "not_monitored"
        if self.lambda_phi_monitor["violations"] > 0.0:
            return "calibration_invalid"
        return "conservative"

    def _alpha_t(self) -> float:
        if self.config.anytime:
            return anytime_alpha_fn(self.config.alpha, self.t)
        return safe_alpha(self.config.alpha)

    def _dense_alpha_t(self) -> float:
        if not self.config.dense_cauchy:
            return 0.0
        return safe_alpha(self._alpha_t() * self.config.dense_cauchy_fraction)

    def _arm_alpha_t(self) -> float:
        if not self.config.dense_cauchy:
            return self._alpha_t()
        return safe_alpha(self._alpha_t() * (1.0 - self.config.dense_cauchy_fraction))

    def _alpha_per_stat(self) -> float:
        active_arms = max(len(self.config.arms), 1)
        return safe_alpha(self._arm_alpha_t() / (active_arms * max(int(self.n_channels or 1), 1)))

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> StepResult:
        self.t += 1
        self.segment_t += 1
        z = as_vector(z_t)
        self._ensure_channels(z.size)
        residual = self.residualizer.update(z, endogenous_delta=endogenous_delta)
        self._maybe_calibrate_lambda_phi(residual)
        self._monitor_lambda_phi(residual)
        alpha_per = self._alpha_per_stat()
        suppress_alarm = self.segment_t <= self.config.warmup

        scale_results: list[ArmResult] = []
        dynamics_results: list[ArmResult] = []
        location_results: list[ArmResult] = []
        alarms: list[Alarm] = []
        statistics: list[Alarm] = []
        dense_p_values: list[float] = []

        for channel, value in enumerate(residual):
            if "scale" in self.config.arms:
                result = self.scale_arms[channel].update(float(value), alpha_per)
                scale_results.append(result)
                if result.p_value is not None:
                    dense_p_values.append(result.p_value)
                if result.ready:
                    candidate = Alarm(
                        time=self.t,
                        channel=channel,
                        arm="scale",
                        statistic=result.statistic,
                        threshold=result.threshold,
                        p_value=result.p_value,
                        metadata=result.metadata,
                    )
                    statistics.append(candidate)
                    if result.fired and not suppress_alarm:
                        alarms.append(candidate)

            if "location" in self.config.arms:
                result = self.location_arms[channel].update(float(value), alpha_per)
                location_results.append(result)
                if result.p_value is not None:
                    dense_p_values.append(result.p_value)
                if result.ready:
                    candidate = Alarm(
                        time=self.t,
                        channel=channel,
                        arm="location",
                        statistic=result.statistic,
                        threshold=result.threshold,
                        p_value=result.p_value,
                        metadata=result.metadata,
                    )
                    statistics.append(candidate)
                    if result.fired and not suppress_alarm:
                        alarms.append(candidate)

            if "anchored_location" in self.config.arms:
                result = self.anchored_location_arms[channel].update(float(value), alpha_per)
                location_results.append(result)
                if result.p_value is not None:
                    dense_p_values.append(result.p_value)
                if result.ready:
                    candidate = Alarm(
                        time=self.t,
                        channel=channel,
                        arm="anchored_location",
                        statistic=result.statistic,
                        threshold=result.threshold,
                        p_value=result.p_value,
                        metadata=result.metadata,
                    )
                    statistics.append(candidate)
                    if result.fired and not suppress_alarm:
                        alarms.append(candidate)

            if "dynamics" in self.config.arms:
                result = self.dynamics_arms[channel].update(float(value), alpha_per)
                dynamics_results.append(result)
                if result.p_value is not None:
                    dense_p_values.append(result.p_value)
                if result.ready:
                    candidate = Alarm(
                        time=self.t,
                        channel=channel,
                        arm="dynamics",
                        statistic=result.statistic,
                        threshold=result.threshold,
                        p_value=result.p_value,
                        metadata=result.metadata,
                    )
                    statistics.append(candidate)
                    if result.fired and not suppress_alarm:
                        alarms.append(candidate)

        if self.config.dense_cauchy and dense_p_values and not suppress_alarm:
            dense_p = cauchy_combination_p_value(dense_p_values)
            dense_alpha = self._dense_alpha_t()
            dense_candidate = Alarm(
                time=self.t,
                channel=None,
                arm="dense_cauchy",
                statistic=1.0 - dense_p,
                threshold=1.0 - dense_alpha,
                p_value=dense_p,
            )
            statistics.append(dense_candidate)
            if dense_p < dense_alpha:
                alarms.append(dense_candidate)

        if alarms:
            self._alarm_count += 1
            if self.config.reset_after_alarm:
                saved = StepResult(
                    self.t,
                    residual.copy(),
                    scale_results,
                    dynamics_results,
                    location_results,
                    alarms,
                    statistics,
                )
                self._reset_segment()
                return saved

        return StepResult(
            self.t,
            residual.copy(),
            scale_results,
            dynamics_results,
            location_results,
            alarms,
            statistics,
        )

    def run(
        self,
        stream: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> Alarm | None:
        stream = np.asarray(stream, dtype=float)
        if stream.ndim == 1:
            stream = stream.reshape(-1, 1)
        deltas = endogenous_delta
        if deltas is not None:
            deltas = np.asarray(deltas, dtype=float)
            if deltas.ndim == 1:
                deltas = deltas.reshape(-1, 1)
        for idx, row in enumerate(stream):
            delta = None if deltas is None else deltas[idx]
            result = self.update(row, endogenous_delta=delta)
            if result.alarm is not None:
                return result.alarm
        return None
