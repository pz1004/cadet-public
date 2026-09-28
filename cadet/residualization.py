"""Policy-update residualization implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from cadet.utils import as_vector


class Residualizer:
    """Base interface for stream residualizers."""

    def reset(self) -> None:
        raise NotImplementedError

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class IdentityResidualizer(Residualizer):
    """No residualization: detectors see the raw monitored stream."""

    def reset(self) -> None:
        return None

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> np.ndarray:
        return as_vector(z_t).copy()


@dataclass
class LinearPURResidualizer(Residualizer):
    """Accumulate caller-supplied per-channel endogenous increments.

    MuJoCo traces supply exact frozen-buffer feature differences (zero for
    parameter-independent readouts). A caller may instead supply a JVP
    approximation; its Taylor remainders accumulate across updates. Neither
    choice establishes stationarity of the monitored residual stream.
    """

    initial_mu: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.mu_: np.ndarray | None = None

    def reset(self) -> None:
        self.mu_ = None if self.initial_mu is None else as_vector(self.initial_mu).copy()

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> np.ndarray:
        z = as_vector(z_t)
        if self.mu_ is None:
            self.mu_ = np.zeros_like(z) if self.initial_mu is None else as_vector(self.initial_mu).copy()
        if endogenous_delta is not None:
            delta = as_vector(endogenous_delta)
            if delta.shape != z.shape:
                raise ValueError(f"endogenous_delta shape {delta.shape} does not match z_t {z.shape}")
            self.mu_ = self.mu_ + delta
        return z - self.mu_


@dataclass
class FrozenBufferPURResidualizer(Residualizer):
    """Frozen-buffer PUR with an optional approximate JVP callback.

    ``feature_fn(buffer, params)`` must evaluate the monitored statistic on the
    frozen state buffer and return either a feature vector or a matrix whose
    first dimension indexes buffer states. ``jvp_fn`` may be supplied by an
    autodiff backend; if omitted, the implementation evaluates the exact feature
    difference at ``params + delta_params`` and ``params``. The historical
    ``finite_difference_eps`` cancels algebraically and does not control a
    small-step derivative approximation.
    """

    feature_fn: Callable[[Any, Any], np.ndarray]
    params: Any
    buffer: Any
    jvp_fn: Callable[[Any, Any, Any], np.ndarray] | None = None
    finite_difference_eps: float = 1e-4
    refresh_every: int | None = None
    refresh_fn: Callable[[int], Any] | None = None

    def __post_init__(self) -> None:
        self.mu_: np.ndarray | None = None
        self.steps_ = 0

    def reset(self) -> None:
        self.mu_ = None
        self.steps_ = 0

    def _mean_feature(self, params: Any) -> np.ndarray:
        values = np.asarray(self.feature_fn(self.buffer, params), dtype=float)
        if values.ndim == 1:
            return values.copy()
        if values.ndim < 1:
            return values.reshape(1)
        return np.mean(values, axis=0)

    def _jvp(self, delta_params: Any) -> np.ndarray:
        if self.jvp_fn is not None:
            return as_vector(self.jvp_fn(self.buffer, self.params, delta_params))
        try:
            moved = self.params + delta_params
        except TypeError as exc:
            raise TypeError(
                "finite-difference PUR requires params and delta_params to support addition; "
                "provide jvp_fn for framework-specific parameter containers"
            ) from exc
        return (self._mean_feature(moved) - self._mean_feature(self.params)) / max(
            self.finite_difference_eps, 1e-12
        ) * self.finite_difference_eps

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | dict[str, Any] | None = None,
    ) -> np.ndarray:
        z = as_vector(z_t)
        if self.mu_ is None:
            self.mu_ = self._mean_feature(self.params)
        if endogenous_delta is None:
            delta_params = None
            new_params = None
        elif isinstance(endogenous_delta, dict):
            delta_params = endogenous_delta.get("delta_params")
            new_params = endogenous_delta.get("new_params")
        else:
            delta_params = endogenous_delta
            new_params = None

        if delta_params is not None:
            self.mu_ = self.mu_ + as_vector(self._jvp(delta_params))
            if new_params is not None:
                self.params = new_params
            else:
                try:
                    self.params = self.params + delta_params
                except TypeError:
                    pass
        elif new_params is not None:
            self.params = new_params

        self.steps_ += 1
        if self.refresh_every and self.refresh_fn and self.steps_ % self.refresh_every == 0:
            self.buffer = self.refresh_fn(self.steps_)
        return z - self.mu_


@dataclass
class TimescaleResidualizer(Residualizer):
    """Model-free high-pass fallback using a slow exponential moving average."""

    rate: float = 0.005

    def __post_init__(self) -> None:
        if not 0.0 < self.rate < 1.0:
            raise ValueError("rate must lie in (0, 1)")
        self.mu_: np.ndarray | None = None

    def reset(self) -> None:
        self.mu_ = None

    def update(
        self,
        z_t: np.ndarray,
        endogenous_delta: np.ndarray | None = None,
    ) -> np.ndarray:
        z = as_vector(z_t)
        if self.mu_ is None:
            self.mu_ = z.copy()
            return np.zeros_like(z)
        residual = z - self.mu_
        self.mu_ = self.mu_ + self.rate * residual
        return residual
