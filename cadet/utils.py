"""Numerical helpers shared by detectors and experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass
class P2Quantile:
    """Jain-Chlamtac P2 streaming quantile estimator."""

    q: float

    def __post_init__(self) -> None:
        if not 0.0 < self.q < 1.0:
            raise ValueError("q must lie in (0, 1)")
        self.initial: list[float] = []
        self.heights: np.ndarray | None = None
        self.positions: np.ndarray | None = None
        self.desired: np.ndarray | None = None
        self.increments = np.asarray([0.0, self.q / 2.0, self.q, (1.0 + self.q) / 2.0, 1.0])

    def update(self, value: float) -> float:
        x = float(value)
        if self.heights is None:
            self.initial.append(x)
            if len(self.initial) == 5:
                self.heights = np.sort(np.asarray(self.initial, dtype=float))
                self.positions = np.arange(1, 6, dtype=float)
                self.desired = np.asarray(
                    [1.0, 1.0 + 2.0 * self.q, 1.0 + 4.0 * self.q, 3.0 + 2.0 * self.q, 5.0],
                    dtype=float,
                )
            return float(np.quantile(self.initial, self.q))

        assert self.positions is not None and self.desired is not None
        h = self.heights
        if x < h[0]:
            h[0] = x
            k = 0
        elif x >= h[4]:
            h[4] = x
            k = 3
        else:
            k = int(np.searchsorted(h, x, side="right") - 1)
            k = max(0, min(k, 3))

        self.positions[k + 1 :] += 1.0
        self.desired += self.increments

        for i in range(1, 4):
            d = self.desired[i] - self.positions[i]
            if (d >= 1.0 and self.positions[i + 1] - self.positions[i] > 1.0) or (
                d <= -1.0 and self.positions[i - 1] - self.positions[i] < -1.0
            ):
                step = float(np.sign(d))
                hp = self._parabolic(i, step)
                if h[i - 1] < hp < h[i + 1]:
                    h[i] = hp
                else:
                    h[i] = self._linear(i, step)
                self.positions[i] += step
        return self.value

    def _parabolic(self, i: int, step: float) -> float:
        assert self.heights is not None and self.positions is not None
        h = self.heights
        n = self.positions
        return float(
            h[i]
            + step
            / (n[i + 1] - n[i - 1])
            * (
                (n[i] - n[i - 1] + step) * (h[i + 1] - h[i]) / (n[i + 1] - n[i])
                + (n[i + 1] - n[i] - step) * (h[i] - h[i - 1]) / (n[i] - n[i - 1])
            )
        )

    def _linear(self, i: int, step: float) -> float:
        assert self.heights is not None and self.positions is not None
        j = i + int(step)
        return float(
            self.heights[i]
            + step * (self.heights[j] - self.heights[i]) / (self.positions[j] - self.positions[i])
        )

    @property
    def value(self) -> float:
        if self.heights is None:
            if not self.initial:
                return 0.0
            return float(np.quantile(self.initial, self.q))
        return float(self.heights[2])


class P2IQR:
    """Constant-memory median/IQR readout."""

    def __init__(self) -> None:
        self.q25 = P2Quantile(0.25)
        self.q50 = P2Quantile(0.50)
        self.q75 = P2Quantile(0.75)
        self.count = 0

    def update(self, value: float) -> None:
        self.q25.update(value)
        self.q50.update(value)
        self.q75.update(value)
        self.count += 1

    @property
    def median(self) -> float:
        return self.q50.value

    @property
    def iqr(self) -> float:
        return float(max(self.q75.value - self.q25.value, EPS))

    def reset(self) -> None:
        self.__init__()


def as_vector(x: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D feature vector, got shape {arr.shape}")
    return arr


def safe_alpha(alpha: float) -> float:
    return float(np.clip(alpha, 1e-300, 1.0 - 1e-12))


def anytime_alpha(alpha: float, t: int) -> float:
    """Polynomial alpha-spending schedule from Theorem 4."""
    t = max(int(t), 1)
    return safe_alpha(alpha * 6.0 / (math.pi**2 * t**2))


def effective_n(n: int, lambda_phi: float = 1.0) -> int:
    """KR-style effective sample size n / Lambda_phi^2."""
    lam = max(float(lambda_phi), 1.0)
    return max(2, int(math.floor(n / (lam * lam))))


def dkw_scale_threshold(alpha: float, n: int, lambda_phi: float = 1.0) -> float:
    """Two-window dispersion-ECDF threshold from Proposition 1."""
    n_eff = effective_n(n, lambda_phi)
    return math.sqrt((2.0 / n_eff) * math.log(4.0 / safe_alpha(alpha)))


def dkw_scale_p_value(statistic: float, n: int, lambda_phi: float = 1.0) -> float:
    n_eff = effective_n(n, lambda_phi)
    return min(1.0, 4.0 * math.exp(-0.5 * n_eff * statistic * statistic))


def dynamics_radius(alpha: float, m: int, lambda_phi: float = 1.0) -> float:
    lam = max(float(lambda_phi), 1.0)
    return lam * math.sqrt((2.0 / max(int(m), 1)) * math.log(8.0 / safe_alpha(alpha)))


def dynamics_p_value(max_abs_deviation: float, m: int, lambda_phi: float = 1.0) -> float:
    lam = max(float(lambda_phi), 1.0)
    exponent = -max(int(m), 1) * max_abs_deviation**2 / (2.0 * lam * lam)
    return min(1.0, 4.0 * math.exp(exponent))


def robust_iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    q75, q25 = np.percentile(x, [75.0, 25.0])
    return float(max(q75 - q25, EPS))


def robust_scale(x: np.ndarray) -> float:
    """IQR converted to a Gaussian-sigma scale, with a MAD fallback."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 1.0
    scale = robust_iqr(x) / 1.349
    if scale <= EPS:
        med = np.median(x)
        scale = 1.4826 * np.median(np.abs(x - med))
    return float(max(scale, EPS))


def winsorize(x: np.ndarray, c: float = 4.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    scale = robust_scale(x)
    return np.clip(x, med - c * scale, med + c * scale)


def dispersion_values(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return np.abs(x - med)


def two_sample_ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sample Kolmogorov distance evaluated at pooled jumps."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def dispersion_ks_distance(reference: np.ndarray, detection: np.ndarray) -> float:
    return two_sample_ks_distance(dispersion_values(reference), dispersion_values(detection))


def flip_rate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    med = np.median(values)
    signs = values >= med
    return float(np.mean(signs[1:] != signs[:-1]))


def robust_lag1_corr(values: np.ndarray) -> float:
    """Winsorized lag-1 correlation used by the robust dynamics GLR baseline."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 0.0
    y = winsorize(values)
    left = y[:-1] - np.mean(y[:-1])
    right = y[1:] - np.mean(y[1:])
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= EPS:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, -0.98, 0.98))


def cauchy_combination_p_value(p_values: list[float]) -> float:
    """Cauchy combination p-value, valid under broad dependence conditions.

    The transform ``tan((0.5 - p) * pi)`` diverges as ``p -> 1``, so a single
    non-informative input dominates the mean and drives the combined value to 1,
    destroying the statistic. That matters here because the arms report *bounds*
    -- ``min(1, 4 exp(-n eps^2 / 2))`` for the scale arm -- which are exactly 1
    whenever the statistic is small, which is most steps. Saturated inputs carry no
    evidence, so they are dropped before combining rather than allowed to swamp it.
    If every input is saturated the result is 1, as it should be.
    """

    if not p_values:
        return 1.0
    values = np.asarray(p_values, dtype=float)
    informative = values[np.isfinite(values) & (values < 1.0)]
    if informative.size == 0:
        return 1.0
    clipped = np.clip(informative, 1e-15, 1.0 - 1e-12)
    stat = np.mean(np.tan((0.5 - clipped) * math.pi))
    return float(np.clip(0.5 - math.atan(stat) / math.pi, 0.0, 1.0))


def multiplicity_corrected_z(tests: int, alpha: float = 0.05) -> float:
    """Two-sided Gaussian cut-off for ``tests`` simultaneous autocorrelation checks."""

    from scipy.stats import norm

    tests = max(int(tests), 1)
    return float(norm.ppf(1.0 - alpha / (2.0 * tests)))


def estimate_lambda_phi(
    values: np.ndarray,
    max_lag: int = 50,
    noise_floor_z: float = 2.0,
) -> float:
    """Legacy absolute-autocorrelation diagnostic, not a certified mixing bound.

    Autocorrelations do not identify phi-mixing coefficients or the
    Kontorovich--Ramanan mixing matrix. Inflating this summary does not by itself
    establish either bound; the historical name is retained for trace replay.

    Sample autocorrelations of an independent series have standard error
    ``1/sqrt(n)``, so lags below ``noise_floor_z / sqrt(n)`` are discarded as noise. The
    default ``z = 2`` is a per-lag two-sigma cut, which is right for a single estimate
    but far too permissive when the estimate is *repeated* -- across ``max_lag`` lags and
    across channels, at least one spurious exceedance is near-certain, which makes a
    monitor built on this estimator flag almost every block of independent data. Callers
    that repeat the test should pass a multiplicity-corrected ``noise_floor_z``; see
    ``multiplicity_corrected_z``.
    """

    x = np.asarray(values, dtype=float)
    if x.size < 4:
        return 1.0
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom <= EPS:
        return 1.0
    noise_floor = float(noise_floor_z) / math.sqrt(x.size)
    acc = 1.0
    for lag in range(1, min(max_lag, x.size - 1) + 1):
        rho = abs(float(np.dot(x[:-lag], x[lag:]) / denom))
        if rho < noise_floor:
            break
        acc += min(max(rho - noise_floor, 0.0), 0.95)
    return float(max(1.0, acc))


def robust_whittle_ar1_fit(
    increments: np.ndarray,
    clip: float = 6.0,
    grid_size: int = 21,
    max_iter: int = 15,
    tol: float = 1e-6,
    return_info: bool = False,
) -> tuple[float, float, float]:
    """Robust Whittle fit for an AR(1) increment model.

    The solver uses a coarse grid only to initialize phi, then refines the
    robust Whittle estimating equations by damped Fisher scoring in constrained
    coordinates ``phi=tanh(eta)``, ``sigma2=exp(xi)``. If scoring fails to
    improve the robust objective, the best grid solution is returned as a
    numerical safeguard.
    """

    x = np.asarray(increments, dtype=float)
    x = x - np.median(x)
    n = x.size
    if n < 8:
        result = (0.0, max(float(np.var(x)), EPS), 0.0)
        if return_info:
            return (*result, {"solver_status": "too_short", "iterations": 0})  # type: ignore[return-value]
        return result

    freqs = 2.0 * math.pi * np.arange(1, n // 2 + 1) / n
    fft = np.fft.rfft(x)
    periodogram = (np.abs(fft[1 : freqs.size + 1]) ** 2) / (2.0 * math.pi * n)
    periodogram = np.maximum(periodogram, EPS)
    target = 1.0 - math.exp(-clip)

    def rho(u: np.ndarray) -> np.ndarray:
        u = np.maximum(u, EPS)
        return np.where(u <= clip, u, clip + clip * np.log(u / clip))

    def objective(phi: float, sigma2: float) -> float:
        sigma2 = max(float(sigma2), EPS)
        denom = np.maximum(1.0 - 2.0 * phi * np.cos(freqs) + phi * phi, EPS)
        u = periodogram * 2.0 * math.pi * denom / sigma2
        log_f = math.log(sigma2) - np.log(2.0 * math.pi * denom)
        return float(np.sum(target * log_f + rho(u)))

    def score_info(eta: float, xi: float) -> tuple[float, np.ndarray, np.ndarray, float, float]:
        phi = float(np.tanh(eta))
        sigma2 = float(np.exp(np.clip(xi, math.log(EPS), 60.0)))
        denom = np.maximum(1.0 - 2.0 * phi * np.cos(freqs) + phi * phi, EPS)
        u = periodogram * 2.0 * math.pi * denom / sigma2
        residual = np.minimum(u, clip) - target
        dlogf_dphi = 2.0 * (np.cos(freqs) - phi) / denom
        a_eta = dlogf_dphi * (1.0 - phi * phi)
        a_xi = np.ones_like(a_eta)
        design = np.column_stack([a_eta, a_xi])
        score = design.T @ residual
        weights = np.where(u < clip, u, 0.0)
        if float(np.sum(weights)) <= EPS:
            weights = np.full_like(u, target)
        info = design.T @ (design * weights[:, None])
        info += np.eye(2) * 1e-8
        return objective(phi, sigma2), score, info, phi, sigma2

    best_phi = 0.0
    best_sigma2 = max(float(np.var(x)), EPS)
    best_obj = float("inf")
    for phi in np.linspace(-0.95, 0.95, grid_size):
        denom = np.maximum(1.0 - 2.0 * phi * np.cos(freqs) + phi * phi, EPS)
        base = periodogram * 2.0 * math.pi * denom
        lo = EPS
        hi = max(float(np.max(base)) / target, EPS)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            clipped_mean = float(np.mean(np.minimum(base / mid, clip)))
            if clipped_mean > target:
                lo = mid
            else:
                hi = mid
        sigma2 = max(0.5 * (lo + hi), EPS)
        obj = objective(float(phi), sigma2)
        if obj < best_obj:
            best_phi = float(phi)
            best_sigma2 = float(sigma2)
            best_obj = obj

    eta = float(np.arctanh(np.clip(best_phi, -0.999, 0.999)))
    xi = float(math.log(max(best_sigma2, EPS)))
    best_refined = (best_phi, best_sigma2, best_obj)
    status = "grid_fallback"
    iterations = 0

    for iterations in range(1, max_iter + 1):
        obj, score, info, phi, sigma2 = score_info(eta, xi)
        if obj < best_refined[2]:
            best_refined = (phi, sigma2, obj)
        if float(np.linalg.norm(score, ord=2)) < tol:
            status = "fisher_converged"
            break
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, score, rcond=None)[0]
        step_norm = float(np.linalg.norm(step, ord=2))
        if not np.isfinite(step_norm) or step_norm <= EPS:
            status = "fisher_stalled"
            break
        if step_norm > 1.5:
            step = step * (1.5 / step_norm)

        accepted = False
        for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            cand_eta = eta + shrink * float(step[0])
            cand_xi = float(np.clip(xi + shrink * float(step[1]), math.log(EPS), 60.0))
            cand_obj, _, _, cand_phi, cand_sigma2 = score_info(cand_eta, cand_xi)
            if np.isfinite(cand_obj) and cand_obj <= obj + 1e-10:
                eta, xi = cand_eta, cand_xi
                if cand_obj < best_refined[2]:
                    best_refined = (cand_phi, cand_sigma2, cand_obj)
                accepted = True
                status = "fisher_refined"
                break
        if not accepted:
            status = "grid_fallback" if best_refined[2] >= best_obj - 1e-10 else "fisher_refined"
            break

    result = best_refined if best_refined[2] <= best_obj else (best_phi, best_sigma2, best_obj)
    if return_info:
        return (  # type: ignore[return-value]
            result[0],
            result[1],
            result[2],
            {
                "solver_status": status,
                "iterations": iterations,
                "grid_phi": best_phi,
                "grid_sigma2": best_sigma2,
                "grid_objective": best_obj,
            },
        )
    return result
