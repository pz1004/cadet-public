"""Synthetic and CSV stream sources for CADET experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


DEFAULT_FEATURE_NAMES = ("td_abs", "value", "entropy", "max_logit", "hidden_norm")


@dataclass
class SimulatedRun:
    z: np.ndarray
    endogenous_delta: np.ndarray
    tau: int | None
    feature_names: tuple[str, ...]
    scenario: str
    # Optional per-step PUR numerical diagnostics; populated only when a trace is
    # generated with ``record_update_diagnostics=True``. Not persisted with the
    # trace arrays, so cached ``.npz`` files are unaffected.
    update_diagnostics: list[dict[str, float]] | None = None


def _scenario_params(
    scenario: str,
    d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.ones(d, dtype=float)
    phi = np.full(d, 0.15, dtype=float)
    mean = np.zeros(d, dtype=float)

    if scenario == "no_shift":
        return scale, phi, mean
    if scenario == "scale":
        scale[0] = 2.8
        if d > 4:
            scale[4] = 2.0
        return scale, phi, mean
    if scenario == "dynamics":
        phi[0] = -0.92
        if d > 1:
            phi[1] = 0.95
        return scale, phi, mean
    if scenario == "mixed":
        scale[0] = 3.0
        if d > 1:
            phi[1] = 0.95
        if d > 2:
            mean[2] = -0.45
        if d > 3:
            mean[3] = 0.35
        if d > 4:
            scale[4] = 2.2
        return scale, phi, mean
    if scenario == "mean":
        mean[0] = 0.9
        if d > 1:
            mean[1] = -0.5
        return scale, phi, mean
    if scenario == "dense":
        # Dense alternative: every channel moves by a small common factor. This is the
        # regime in which a pooled or Cauchy-combined statistic is expected to beat
        # per-channel Bonferroni min-p, and it is the counterpart to the concentrated
        # "scale" scenario in which one channel carries the whole shift.
        scale[:] = 1.35
        return scale, phi, mean
    if scenario == "dense_mean":
        mean[:] = 0.22
        return scale, phi, mean
    raise ValueError(f"unknown scenario {scenario!r}")


def generate_rl_internal_stream(
    horizon: int = 1200,
    tau: int | None = 600,
    d: int = 4,
    scenario: str = "mixed",
    seed: int | None = None,
    endogenous_step: float = 0.035,
    innovation_scale: float = 0.6,
    df: float = 4.0,
    post_scale_override: np.ndarray | None = None,
) -> SimulatedRun:
    """Generate an RL-like internal feature stream with known PUR increments.

    The stream is ``z_t = mu_t^endo + r_t``. ``endogenous_delta[t]`` is exactly
    ``mu_t^endo - mu_{t-1}^endo`` with ``endogenous_delta[0] = mu_0^endo``, so
    a LinearPURResidualizer can remove the endogenous drift without knowing the
    simulator internals.
    """

    rng = np.random.default_rng(seed)
    if scenario == "no_shift":
        tau = None
    if tau is None:
        change_idx = horizon + 1
    else:
        change_idx = int(tau)

    feature_names = tuple(DEFAULT_FEATURE_NAMES[:d])
    if len(feature_names) < d:
        feature_names = feature_names + tuple(f"feature_{i}" for i in range(len(feature_names), d))

    raw_steps = endogenous_step * rng.normal(size=(horizon, d))
    seasonal = 0.015 * np.sin(np.linspace(0.0, 8.0 * np.pi, horizon)).reshape(-1, 1)
    raw_steps = raw_steps + seasonal * rng.normal(size=(1, d))
    endogenous_mu = np.cumsum(raw_steps, axis=0)
    endogenous_delta = np.empty_like(endogenous_mu)
    endogenous_delta[0] = endogenous_mu[0]
    endogenous_delta[1:] = np.diff(endogenous_mu, axis=0)

    post_scale, post_phi, post_mean = _scenario_params(scenario, d)
    if post_scale_override is not None:
        # Explicit per-channel post-change scale factors. Used by the fusion study to
        # spread a *fixed total* signal over a varying number of channels, which is the
        # only construction under which concentrated and dense alternatives are
        # comparable rather than merely different in magnitude.
        override = np.asarray(post_scale_override, dtype=float).reshape(-1)
        if override.size != d:
            raise ValueError(f"post_scale_override must have length {d}, got {override.size}")
        post_scale = override
    pre_scale = np.ones(d, dtype=float)
    pre_phi = np.full(d, 0.15, dtype=float)
    pre_mean = np.zeros(d, dtype=float)

    residual = np.zeros((horizon, d), dtype=float)
    latent_previous = np.zeros(d, dtype=float)
    for t in range(horizon):
        changed = t >= change_idx
        scale = post_scale if changed else pre_scale
        phi = post_phi if changed else pre_phi
        mean = post_mean if changed else pre_mean
        # Gaussian-copula dynamics with Student-t marginals. Changing phi
        # changes temporal dependence while preserving each channel's marginal
        # law, which makes the dynamics-only scenario match the paper's
        # complementarity construction.
        latent = phi * latent_previous + np.sqrt(np.maximum(1.0 - phi * phi, 1e-6)) * rng.normal(size=d)
        u = np.clip(norm.cdf(latent), 1e-9, 1.0 - 1e-9)
        marginal = student_t.ppf(u, df=df) / np.sqrt(df / (df - 2.0))
        current = mean + innovation_scale * scale * marginal
        residual[t] = current
        latent_previous = latent

    z = endogenous_mu + residual
    return SimulatedRun(
        z=z,
        endogenous_delta=endogenous_delta,
        tau=tau,
        feature_names=feature_names,
        scenario=scenario,
    )


def load_csv_stream(
    path: str | Path,
    feature_columns: list[str],
    delta_columns: list[str] | None = None,
    tau: int | None = None,
    scenario: str = "csv",
) -> SimulatedRun:
    """Load a feature stream exported from an RL run."""

    frame = pd.read_csv(path)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature column(s): {missing}")
    z = frame[feature_columns].to_numpy(dtype=float)

    if delta_columns is None:
        endogenous_delta = np.zeros_like(z)
    else:
        missing_delta = [column for column in delta_columns if column not in frame.columns]
        if missing_delta:
            raise ValueError(f"missing delta column(s): {missing_delta}")
        endogenous_delta = frame[delta_columns].to_numpy(dtype=float)

    return SimulatedRun(
        z=z,
        endogenous_delta=endogenous_delta,
        tau=tau,
        feature_names=tuple(feature_columns),
        scenario=scenario,
    )
