"""Empirical statistic-threshold calibration for detector comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import beta

from cadet.evaluation import (
    ExperimentConfig,
    default_detector_factories,
    generate_experiment_stream,
)
from cadet.simulation import SimulatedRun


NO_ALARM_ALPHA = 1e-300


@dataclass(frozen=True)
class EmpiricalCalibrationResult:
    """Tables produced by empirical run-level false-alarm calibration."""

    raw_statistics: pd.DataFrame
    run_maxima: pd.DataFrame
    thresholds: pd.DataFrame
    heldout_far: pd.DataFrame
    detector_status: pd.DataFrame


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute an exact binomial confidence interval.

    Args:
        successes: Number of positive Bernoulli outcomes.
        trials: Number of Bernoulli trials.
        confidence: Desired two-sided confidence level.

    Returns:
        Lower and upper Clopper-Pearson interval endpoints.
    """

    if trials <= 0:
        return float("nan"), float("nan")
    if not 0 <= successes <= trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    alpha = 1.0 - float(confidence)
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


def clopper_pearson_upper_bound(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    """One-sided exact binomial upper confidence bound.

    A zero-count point estimate from a small sample is compatible with a true rate
    well above the nominal target, so every reported ``0/n`` false-alarm count is
    accompanied by this bound. ``clopper_pearson_upper_bound(0, 20)`` is roughly
    ``0.139``: 0/20 no-shift alarms does not by itself certify a rate below 0.05.

    Args:
        successes: Number of positive Bernoulli outcomes.
        trials: Number of Bernoulli trials.
        confidence: One-sided confidence level.

    Returns:
        The one-sided upper endpoint, or NaN when ``trials`` is non-positive.
    """

    if trials <= 0:
        return float("nan")
    if not 0 <= successes <= trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    if successes == trials:
        return 1.0
    alpha = 1.0 - float(confidence)
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def empirical_run_threshold(maxima: Sequence[float], target_far: float) -> float:
    """Select a conservative run-level empirical threshold.

    The threshold is the ``1 - target_far`` empirical quantile using the nearest
    higher observed maximum. Evaluation uses a strict ``>`` comparison, so this
    choice is conservative on the calibration split.
    """

    values = np.asarray([value for value in maxima if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return float("inf")
    if not 0.0 <= target_far <= 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {target_far}")
    quantile = float(np.clip(1.0 - target_far, 0.0, 1.0))
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, quantile, interpolation="higher"))


def experiment_seeds(config: ExperimentConfig, runs: int | None = None) -> tuple[int, ...]:
    """Return the deterministic no-shift seed schedule used for calibration."""

    n_runs = int(runs if runs is not None else (config.eval_runs or config.n_runs))
    return tuple(int(config.seed + 1009 * run_idx) for run_idx in range(n_runs))


def _env_ids(config: ExperimentConfig) -> tuple[str, ...]:
    if config.source == "mujoco":
        return tuple(config.mujoco_env_ids) if config.mujoco_env_ids else (config.mujoco_env_id,)
    return ("synthetic",)


def _scenario_seed(config: ExperimentConfig, run_idx: int, scenario: str, env_index: int) -> int:
    scenario_offsets = {
        "no_shift": 0,
        "scale": 1,
        "dynamics": 2,
        "mixed": 3,
        "mean": 4,
        "gravity": 5,
        "friction": 6,
        "target": 7,
        "sensor_bias": 8,
        "dense": 10,
        "dense_mean": 11,
    }
    return int(
        config.seed
        + 1009 * run_idx
        + 9173 * scenario_offsets.get(scenario, 99)
        + 104729 * env_index
    )


def _no_shift_stream_records(
    config: ExperimentConfig,
    runs: int | None = None,
) -> tuple[tuple[str, int, int, SimulatedRun], ...]:
    n_runs = int(runs if runs is not None else (config.eval_runs or config.n_runs))
    records: list[tuple[str, int, int, SimulatedRun]] = []
    for env_index, env_id in enumerate(_env_ids(config)):
        for run_idx in range(n_runs):
            seed = _scenario_seed(config, run_idx, "no_shift", env_index)
            records.append(
                (
                    env_id,
                    run_idx,
                    seed,
                    generate_experiment_stream(config, "no_shift", seed, env_id=env_id),
                )
            )
    return tuple(records)


def no_shift_streams(
    config: ExperimentConfig,
    runs: int | None = None,
) -> tuple[SimulatedRun, ...]:
    """Generate no-shift streams for one calibration/evaluation split."""

    return tuple(record[3] for record in _no_shift_stream_records(config, runs))


def _include_references_for(detector_name: str) -> bool:
    return detector_name in {"ScaleGLR", "DynamicsGLR"}


def _detector_factory(config: ExperimentConfig, detector_name: str):
    factories = default_detector_factories(
        config,
        {detector_name: NO_ALARM_ALPHA},
        include_references=_include_references_for(detector_name),
    )
    if detector_name not in factories:
        valid = ", ".join(sorted(factories))
        raise ValueError(f"unknown detector {detector_name!r}; valid detectors: {valid}")
    return factories[detector_name]


def statistic_rows_for_run(
    config: ExperimentConfig,
    detector_name: str,
    stream: SimulatedRun,
    *,
    split: str,
    run_idx: int,
    seed: int,
    env_id: str = "synthetic",
) -> list[dict[str, Any]]:
    """Collect all non-firing candidate statistics for one detector and stream."""

    detector = _detector_factory(config, detector_name)(stream.z.shape[1])
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(stream.z):
        result = detector.update(row, endogenous_delta=stream.endogenous_delta[idx])
        for candidate in result.statistics:
            if int(candidate.time) <= int(config.warmup):
                continue
            rows.append(
                {
                    "split": split,
                    "detector": detector_name,
                    "env_id": env_id,
                    "scenario": stream.scenario,
                    "run": int(run_idx),
                    "seed": int(seed),
                    "time": int(candidate.time),
                    "channel": candidate.channel,
                    "arm": candidate.arm,
                    "statistic": float(candidate.statistic),
                    "nominal_threshold": float(candidate.threshold),
                    "p_value": candidate.p_value,
                }
            )
    return rows


def _run_maxima(statistics: pd.DataFrame) -> pd.DataFrame:
    if statistics.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "detector",
                "env_id",
                "scenario",
                "run",
                "seed",
                "max_statistic",
                "argmax_time",
                "argmax_channel",
            ]
        )

    rows: list[dict[str, Any]] = []
    group_columns = ["split", "detector", "env_id", "scenario", "run", "seed"]
    for key, group in statistics.groupby(group_columns, sort=True):
        idx = group["statistic"].astype(float).idxmax()
        best = group.loc[idx]
        split, detector, env_id, scenario, run, seed = key
        rows.append(
            {
                "split": split,
                "detector": detector,
                "env_id": env_id,
                "scenario": scenario,
                "run": int(run),
                "seed": int(seed),
                "max_statistic": float(best["statistic"]),
                "argmax_time": int(best["time"]),
                "argmax_channel": int(best["channel"]) if pd.notna(best["channel"]) else None,
            }
        )
    return pd.DataFrame(rows)


def collect_no_shift_statistics(
    config: ExperimentConfig,
    detector_names: Sequence[str],
    *,
    split: str,
    runs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect raw statistic logs and run-level maxima on no-shift streams."""

    records = _no_shift_stream_records(config, runs)
    rows: list[dict[str, Any]] = []
    for detector_name in detector_names:
        for env_id, run_idx, seed, stream in records:
            rows.extend(
                statistic_rows_for_run(
                    config,
                    detector_name,
                    stream,
                    split=split,
                    run_idx=run_idx,
                    seed=seed,
                    env_id=env_id,
                )
            )
    statistics = pd.DataFrame(rows)
    return statistics, _run_maxima(statistics)


def calibrate_from_run_maxima(
    calibration_maxima: pd.DataFrame,
    detector_names: Sequence[str],
    target_far: float,
    *,
    group_by_env: bool = False,
) -> pd.DataFrame:
    """Create empirical thresholds from calibration split run maxima."""

    rows: list[dict[str, Any]] = []
    env_ids: tuple[str | None, ...]
    if group_by_env:
        if "env_id" not in calibration_maxima.columns:
            raise ValueError("group_by_env=True requires an env_id column")
        env_ids = tuple(str(env_id) for env_id in sorted(calibration_maxima["env_id"].unique()))
    else:
        env_ids = (None,)

    for detector_name in detector_names:
        for env_id in env_ids:
            group = calibration_maxima[calibration_maxima["detector"] == detector_name]
            if env_id is not None:
                group = group[group["env_id"].astype(str) == env_id]
            values = (
                group["max_statistic"].to_numpy(dtype=float)
                if not group.empty
                else np.asarray([])
            )
            threshold = empirical_run_threshold(values, target_far)
            calibration_false_alarms = int(np.sum(values > threshold)) if values.size else 0
            calibration_far = (
                float(calibration_false_alarms / values.size) if values.size else float("nan")
            )
            row: dict[str, Any] = {
                "detector": detector_name,
                "empirical_threshold": threshold,
                "target_far": float(target_far),
                "calibration_runs": int(values.size),
                "calibration_false_alarms": calibration_false_alarms,
                "calibration_far": calibration_far,
                "calibration_status": "ok" if values.size else "no_statistics",
            }
            if env_id is not None:
                row["env_id"] = env_id
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_heldout_far(
    heldout_maxima: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    target_far: float,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate held-out no-shift FAR and assign detector status labels."""

    rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for _, threshold_row in thresholds.iterrows():
        detector_name = str(threshold_row["detector"])
        threshold = float(threshold_row["empirical_threshold"])
        group = heldout_maxima[heldout_maxima["detector"] == detector_name]
        env_id = None
        if "env_id" in thresholds.columns and pd.notna(threshold_row.get("env_id")):
            env_id = str(threshold_row["env_id"])
            group = group[group["env_id"].astype(str) == env_id]
        values = (
            group["max_statistic"].to_numpy(dtype=float)
            if not group.empty
            else np.asarray([])
        )
        false_alarms = int(np.sum(values > float(threshold))) if values.size else 0
        runs = int(values.size)
        far = float(false_alarms / runs) if runs else float("nan")
        lower, upper = clopper_pearson_interval(false_alarms, runs, confidence)
        if not runs or not np.isfinite(float(threshold)):
            status = "excluded"
            reason = "no held-out statistics"
        elif far <= target_far:
            status = "admissible"
            reason = "held-out point FAR within target"
        else:
            status = "diagnostic_only"
            reason = "held-out point FAR exceeds target"
        result_row: dict[str, Any] = {
            "detector": detector_name,
            "empirical_threshold": float(threshold),
            "target_far": float(target_far),
            "heldout_runs": runs,
            "heldout_false_alarms": false_alarms,
            "heldout_far": far,
            "far_ci_lower": lower,
            "far_ci_upper": upper,
            "confidence": float(confidence),
        }
        status_row: dict[str, Any] = {
            "detector": detector_name,
            "status": status,
            "reason": reason,
            "heldout_far": far,
            "far_ci_lower": lower,
            "far_ci_upper": upper,
        }
        if env_id is not None:
            result_row["env_id"] = env_id
            status_row["env_id"] = env_id
        rows.append(result_row)
        status_rows.append(status_row)
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def run_empirical_calibration(
    calibration_config: ExperimentConfig,
    heldout_config: ExperimentConfig,
    detector_names: Sequence[str],
    *,
    target_far: float,
    confidence: float = 0.95,
    group_by_env: bool = False,
) -> EmpiricalCalibrationResult:
    """Run empirical threshold calibration and held-out FAR evaluation."""

    calibration_stats, calibration_maxima = collect_no_shift_statistics(
        calibration_config,
        detector_names,
        split="calibration",
        runs=calibration_config.calibration_runs,
    )
    heldout_stats, heldout_maxima = collect_no_shift_statistics(
        heldout_config,
        detector_names,
        split="heldout",
        runs=heldout_config.eval_runs or heldout_config.n_runs,
    )
    raw_statistics = pd.concat([calibration_stats, heldout_stats], ignore_index=True)
    run_maxima = pd.concat([calibration_maxima, heldout_maxima], ignore_index=True)
    thresholds = calibrate_from_run_maxima(
        calibration_maxima,
        detector_names,
        target_far,
        group_by_env=group_by_env,
    )
    heldout_far, detector_status = evaluate_heldout_far(
        heldout_maxima,
        thresholds,
        target_far=target_far,
        confidence=confidence,
    )
    return EmpiricalCalibrationResult(
        raw_statistics=raw_statistics,
        run_maxima=run_maxima,
        thresholds=thresholds,
        heldout_far=heldout_far,
        detector_status=detector_status,
    )


def write_empirical_calibration_result(
    result: EmpiricalCalibrationResult,
    outdir: str | Path,
) -> None:
    """Persist empirical calibration artifacts."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result.raw_statistics.to_csv(out / "raw_statistic_log.csv", index=False)
    result.run_maxima.to_csv(out / "run_max_statistics.csv", index=False)
    result.thresholds.to_csv(out / "empirical_thresholds.csv", index=False)
    result.heldout_far.to_csv(out / "heldout_far.csv", index=False)
    result.detector_status.to_csv(out / "detector_status.csv", index=False)


def run_detector_with_empirical_threshold(
    config: ExperimentConfig,
    detector_name: str,
    stream: SimulatedRun,
    threshold: float,
) -> tuple[int | None, float]:
    """Run one detector using an external empirical statistic threshold.

    Args:
        config: Experiment configuration used to build the detector.
        detector_name: Baseline detector name.
        stream: Feature stream to evaluate.
        threshold: Empirical run-level threshold.

    Returns:
        The first empirical-threshold alarm time, if any, and the maximum statistic
        observed over the stream.
    """

    detector = _detector_factory(config, detector_name)(stream.z.shape[1])
    alarm_time: int | None = None
    max_statistic = float("-inf")
    for idx, row in enumerate(stream.z):
        result = detector.update(row, endogenous_delta=stream.endogenous_delta[idx])
        for candidate in result.statistics:
            if int(candidate.time) <= int(config.warmup):
                continue
            statistic = float(candidate.statistic)
            max_statistic = max(max_statistic, statistic)
            if alarm_time is None and statistic > threshold:
                alarm_time = int(candidate.time)
    if not np.isfinite(max_statistic):
        max_statistic = float("nan")
    return alarm_time, max_statistic


def evaluate_empirical_threshold_suite(
    config: ExperimentConfig,
    scenarios: Sequence[str],
    thresholds: pd.DataFrame,
    *,
    detector_names: Sequence[str] | None = None,
    audit_callback: Callable[[SimulatedRun, str, int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Evaluate empirically calibrated baseline detectors on multiple scenarios."""

    has_env_thresholds = "env_id" in thresholds.columns
    threshold_by_detector = dict(
        zip(thresholds["detector"], thresholds["empirical_threshold"], strict=True)
    )
    threshold_by_detector_env = {
        (str(row["detector"]), str(row["env_id"])): float(row["empirical_threshold"])
        for _, row in thresholds.iterrows()
        if has_env_thresholds and pd.notna(row.get("env_id"))
    }
    names = tuple(detector_names or thresholds["detector"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    n_runs = int(config.eval_runs if config.eval_runs is not None else config.n_runs)
    for env_index, env_id in enumerate(_env_ids(config)):
        for scenario in scenarios:
            for run_idx in range(n_runs):
                seed = _scenario_seed(config, run_idx, scenario, env_index)
                stream = generate_experiment_stream(config, scenario, seed, env_id=env_id)
                if audit_callback is not None:
                    audit_callback(stream, scenario, int(run_idx), int(seed), env_id)
                for detector_name in names:
                    if has_env_thresholds:
                        key = (str(detector_name), str(env_id))
                        if key not in threshold_by_detector_env:
                            continue
                        threshold = threshold_by_detector_env[key]
                    else:
                        threshold = float(threshold_by_detector[detector_name])
                    alarm_time, max_statistic = run_detector_with_empirical_threshold(
                        config,
                        detector_name,
                        stream,
                        threshold,
                    )
                    tau = stream.tau
                    has_change = tau is not None
                    false_alarm = alarm_time is not None and (
                        not has_change or alarm_time < int(tau)
                    )
                    detected = alarm_time is not None and has_change and alarm_time >= int(tau)
                    if has_change:
                        delay = (
                            (alarm_time - int(tau))
                            if detected
                            else (stream.z.shape[0] - int(tau))
                        )
                    else:
                        delay = np.nan
                    rows.append(
                        {
                            "env_id": env_id,
                            "scenario": scenario,
                            "run": int(run_idx),
                            "seed": seed,
                            "detector": detector_name,
                            "threshold_mode": "empirical_statistic",
                            "empirical_threshold": threshold,
                            "max_statistic": max_statistic,
                            "tau": tau,
                            "alarm_time": alarm_time,
                            "has_change": has_change,
                            "false_alarm": bool(false_alarm),
                            "detected": bool(detected),
                            "delay": delay,
                            "run_length": int(stream.z.shape[0]),
                            "censored": bool(has_change and not detected and not false_alarm),
                        }
                    )
    return pd.DataFrame(rows)
