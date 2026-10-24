#!/usr/bin/env python
"""Experiment #7: calibration-screened CADET replay on saved MuJoCo traces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadet.calibration import clopper_pearson_interval, empirical_run_threshold  # noqa: E402
from cadet.detectors import CADETConfig, CADETDetector  # noqa: E402
from cadet.evaluation import (  # noqa: E402
    ExperimentConfig,
    default_detector_factories,
    generate_experiment_stream,
    summarize_results,
)
from cadet.residualization import LinearPURResidualizer  # noqa: E402
from cadet.simulation import SimulatedRun  # noqa: E402


SCENARIO_OFFSETS = {
    "no_shift": 0,
    "scale": 1,
    "dynamics": 2,
    "mixed": 3,
    "mean": 4,
    "gravity": 5,
    "friction": 6,
    "target": 7,
    "sensor_bias": 8,
    "actuator_loss": 9,
}

SCREENED_BASELINE_CHOICES = (
    "ADWIN",
    "BOCPD",
    "CUSUM",
    "Energy",
    "MMD",
    "Moment",
    "PageHinkley",
)


@dataclass(frozen=True)
class ScreenedReplaySettings:
    """Settings for calibration-screened replay."""

    outdir: Path
    load_traces: Path | None
    save_traces: Path | None
    mujoco_envs: tuple[str, ...]
    scenarios: tuple[str, ...]
    target_far: float = 0.05
    horizon: int = 600
    tau: int = 300
    calibration_runs: int = 20
    heldout_runs: int = 20
    eval_runs: int = 20
    scale_window: int = 20
    dynamics_window: int = 40
    warmup: int = 80
    mujoco_feature_mode: str = "physics_residual_probe"
    mujoco_policy_std: float = 0.05
    mujoco_policy_mode: str = "linear_actor"
    mujoco_probe_update_mode: str = "online"
    mujoco_frozen_buffer_size: int = 64
    gravity_scale: float = 1.45
    friction_scale: float = 1.8
    sensor_bias_scale: float = 0.35
    actuator_loss_scale: float = 0.55
    arms: tuple[str, ...] = ("scale", "dynamics")
    dense_cauchy: bool = False
    dense_cauchy_fraction: float = 0.5
    screened_baselines: tuple[str, ...] = ()
    screen_max_statistic: float = 0.5
    screen_include_regex: tuple[str, ...] = ()
    screen_exclude_regex: tuple[str, ...] = ()
    calibration_seed: int = 20280620
    heldout_seed: int = 20285620
    evaluation_seed: int = 20290620


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay or generate MuJoCo traces with a calibration-only channel screen. "
            "The screen removes channels whose single-channel no-shift CADET "
            "statistic exceeds a fixed stability cap on the calibration split."
        )
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--load-traces",
        type=Path,
        default=None,
        help="Optional existing trace directory. Missing traces are regenerated only if --save-traces is also set.",
    )
    parser.add_argument(
        "--save-traces",
        type=Path,
        default=None,
        help="Optional trace directory for generated traces; also used as a cache when --load-traces is omitted.",
    )
    parser.add_argument("--mujoco-envs", nargs="+", required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["no_shift", "gravity", "friction"],
        choices=["no_shift", "gravity", "friction", "target", "sensor_bias", "actuator_loss"],
    )
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--tau", type=int, default=300)
    parser.add_argument("--calibration-runs", type=int, default=20)
    parser.add_argument("--heldout-runs", type=int, default=20)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--scale-window", type=int, default=20)
    parser.add_argument("--dynamics-window", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--mujoco-feature-mode", default="physics_residual_probe")
    parser.add_argument("--mujoco-policy-std", type=float, default=0.05)
    parser.add_argument(
        "--mujoco-policy-mode",
        choices=[
            "linear_actor",
            "linear_sinusoidal_dither",
            "linear_probe_burst",
            "zero",
            "sinusoidal",
            "chirp",
            "pulse",
        ],
        default="linear_actor",
    )
    parser.add_argument(
        "--mujoco-probe-update-mode",
        choices=["online", "frozen_policy"],
        default="online",
    )
    parser.add_argument("--mujoco-frozen-buffer-size", type=int, default=64)
    parser.add_argument("--gravity-scale", type=float, default=1.45)
    parser.add_argument("--friction-scale", type=float, default=1.8)
    parser.add_argument("--mujoco-sensor-bias-scale", type=float, default=0.35)
    parser.add_argument("--mujoco-actuator-loss-scale", type=float, default=0.55)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["scale", "dynamics", "location", "anchored_location"],
        default=["scale", "dynamics"],
    )
    parser.add_argument(
        "--dense-cauchy",
        action="store_true",
        help="Enable dense Cauchy fusion of active CADET arm p-values.",
    )
    parser.add_argument(
        "--dense-cauchy-fraction",
        type=float,
        default=0.5,
        help="Fraction of CADET alpha budget reserved for dense Cauchy fusion.",
    )
    parser.add_argument(
        "--screened-baselines",
        nargs="+",
        choices=SCREENED_BASELINE_CHOICES,
        default=[],
        help=(
            "Optional baseline detectors to replay on the same screened channels "
            "with their own empirical run-level thresholds."
        ),
    )
    parser.add_argument(
        "--screen-max-statistic",
        type=float,
        default=0.5,
        help="Keep channels only if their calibration single-channel max statistic is below this cap.",
    )
    parser.add_argument(
        "--screen-include-regex",
        nargs="+",
        default=[],
        help="Optional regex allow-list over feature names before calibration-cap screening.",
    )
    parser.add_argument(
        "--screen-exclude-regex",
        nargs="+",
        default=[],
        help="Optional regex deny-list over feature names before calibration-cap screening.",
    )
    parser.add_argument("--calibration-seed", type=int, default=20280620)
    parser.add_argument("--heldout-seed", type=int, default=20285620)
    parser.add_argument("--evaluation-seed", type=int, default=20290620)
    return parser


def settings_from_args(args: argparse.Namespace) -> ScreenedReplaySettings:
    settings = ScreenedReplaySettings(
        outdir=args.outdir,
        load_traces=args.load_traces,
        save_traces=args.save_traces,
        mujoco_envs=tuple(args.mujoco_envs),
        scenarios=tuple(args.scenarios),
        target_far=float(args.target_far),
        horizon=int(args.horizon),
        tau=int(args.tau),
        calibration_runs=int(args.calibration_runs),
        heldout_runs=int(args.heldout_runs),
        eval_runs=int(args.eval_runs),
        scale_window=int(args.scale_window),
        dynamics_window=int(args.dynamics_window),
        warmup=int(args.warmup),
        mujoco_feature_mode=str(args.mujoco_feature_mode),
        mujoco_policy_std=float(args.mujoco_policy_std),
        mujoco_policy_mode=str(args.mujoco_policy_mode),
        mujoco_probe_update_mode=str(args.mujoco_probe_update_mode),
        mujoco_frozen_buffer_size=int(args.mujoco_frozen_buffer_size),
        gravity_scale=float(args.gravity_scale),
        friction_scale=float(args.friction_scale),
        sensor_bias_scale=float(args.mujoco_sensor_bias_scale),
        actuator_loss_scale=float(args.mujoco_actuator_loss_scale),
        arms=tuple(args.arms),
        dense_cauchy=bool(args.dense_cauchy),
        dense_cauchy_fraction=float(args.dense_cauchy_fraction),
        screened_baselines=tuple(args.screened_baselines),
        screen_max_statistic=float(args.screen_max_statistic),
        screen_include_regex=tuple(args.screen_include_regex),
        screen_exclude_regex=tuple(args.screen_exclude_regex),
        calibration_seed=int(args.calibration_seed),
        heldout_seed=int(args.heldout_seed),
        evaluation_seed=int(args.evaluation_seed),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: ScreenedReplaySettings) -> None:
    if settings.load_traces is None and settings.save_traces is None:
        raise ValueError("one of --load-traces or --save-traces is required")
    if (
        settings.load_traces is not None
        and not settings.load_traces.exists()
        and settings.save_traces is None
    ):
        raise ValueError(f"load-traces directory does not exist: {settings.load_traces}")
    if not settings.mujoco_envs:
        raise ValueError("at least one MuJoCo environment is required")
    if "no_shift" not in settings.scenarios:
        raise ValueError("scenarios must include no_shift")
    if not 0.0 <= settings.target_far <= 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {settings.target_far}")
    if settings.horizon < settings.warmup + settings.scale_window:
        raise ValueError("horizon must be at least warmup + scale_window")
    if settings.horizon < settings.warmup + settings.dynamics_window:
        raise ValueError("horizon must be at least warmup + dynamics_window")
    if settings.screen_max_statistic < 0.0:
        raise ValueError("screen-max-statistic must be non-negative")
    if settings.sensor_bias_scale <= 0.0:
        raise ValueError(
            f"mujoco-sensor-bias-scale must be positive, got {settings.sensor_bias_scale}"
        )
    if not 0.0 < settings.actuator_loss_scale <= 1.0:
        raise ValueError(
            "mujoco-actuator-loss-scale must be in (0, 1], "
            f"got {settings.actuator_loss_scale}"
        )
    if any(run_count <= 0 for run_count in (settings.calibration_runs, settings.heldout_runs, settings.eval_runs)):
        raise ValueError("run counts must be positive")
    if not set(settings.arms) <= {"scale", "dynamics", "location", "anchored_location"}:
        raise ValueError(f"unknown CADET arms: {settings.arms}")
    if not 0.0 < settings.dense_cauchy_fraction < 1.0:
        raise ValueError(
            "dense-cauchy-fraction must be in (0, 1), "
            f"got {settings.dense_cauchy_fraction}"
        )
    unknown_baselines = sorted(set(settings.screened_baselines) - set(SCREENED_BASELINE_CHOICES))
    if unknown_baselines:
        raise ValueError(f"unknown screened baselines: {unknown_baselines}")
    for pattern in (*settings.screen_include_regex, *settings.screen_exclude_regex):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid channel-screen regex {pattern!r}: {exc}") from exc


def channel_name_allowed(
    feature_name: str,
    *,
    include_regex: Sequence[str] = (),
    exclude_regex: Sequence[str] = (),
) -> bool:
    """Return whether a feature name is allowed by optional regex filters."""

    if include_regex and not any(re.search(pattern, feature_name) for pattern in include_regex):
        return False
    if exclude_regex and any(re.search(pattern, feature_name) for pattern in exclude_regex):
        return False
    return True


def scenario_seed(base_seed: int, run_idx: int, scenario: str, env_index: int) -> int:
    return int(
        base_seed
        + 1009 * int(run_idx)
        + 9173 * SCENARIO_OFFSETS.get(scenario, 99)
        + 104729 * int(env_index)
    )


def build_config(settings: ScreenedReplaySettings, seed: int, runs: int) -> ExperimentConfig:
    return ExperimentConfig(
        alpha=settings.target_far,
        horizon=settings.horizon,
        tau=settings.tau,
        n_runs=runs,
        eval_runs=runs,
        calibration_runs=runs,
        scale_window=settings.scale_window,
        dynamics_window=settings.dynamics_window,
        warmup=settings.warmup,
        anytime=True,
        seed=seed,
        source="mujoco",
        mujoco_env_id=settings.mujoco_envs[0],
        mujoco_env_ids=settings.mujoco_envs,
        mujoco_gravity_scale=settings.gravity_scale,
        mujoco_friction_scale=settings.friction_scale,
        mujoco_sensor_bias_scale=settings.sensor_bias_scale,
        mujoco_actuator_loss_scale=settings.actuator_loss_scale,
        mujoco_policy_std=settings.mujoco_policy_std,
        mujoco_policy_mode=settings.mujoco_policy_mode,
        mujoco_feature_mode=settings.mujoco_feature_mode,
        mujoco_probe_update_mode=settings.mujoco_probe_update_mode,
        mujoco_frozen_buffer_size=settings.mujoco_frozen_buffer_size,
        load_traces=None if settings.load_traces is None else str(settings.load_traces),
        save_traces=None if settings.save_traces is None else str(settings.save_traces),
    )


def masked_stream(stream: SimulatedRun, channels: Sequence[int]) -> SimulatedRun:
    idx = np.asarray(tuple(channels), dtype=int)
    return SimulatedRun(
        z=stream.z[:, idx],
        endogenous_delta=stream.endogenous_delta[:, idx],
        tau=stream.tau,
        feature_names=tuple(stream.feature_names[int(i)] for i in idx),
        scenario=stream.scenario,
    )


def cadet_detector(
    config: ExperimentConfig,
    arms: Sequence[str],
    *,
    dense_cauchy: bool = False,
    dense_cauchy_fraction: float = 0.5,
) -> CADETDetector:
    return CADETDetector(
        CADETConfig(
            alpha=config.alpha,
            scale_window=config.scale_window,
            dynamics_window=config.dynamics_window,
            warmup=config.warmup,
            anytime=config.anytime,
            arms=tuple(arms),
            dense_cauchy=bool(dense_cauchy),
            dense_cauchy_fraction=float(dense_cauchy_fraction),
            center_mode=config.center_mode,
            lambda_phi_mode=config.lambda_phi_mode,
            reference_center_warmup=config.reference_center_warmup,
        ),
        residualizer=LinearPURResidualizer(),
    )


def run_stream(
    stream: SimulatedRun,
    config: ExperimentConfig,
    arms: Sequence[str],
    *,
    dense_cauchy: bool = False,
    dense_cauchy_fraction: float = 0.5,
    threshold: float | None = None,
) -> dict[str, object]:
    detector = cadet_detector(
        config,
        arms,
        dense_cauchy=dense_cauchy,
        dense_cauchy_fraction=dense_cauchy_fraction,
    )
    return run_detector_stream(stream, detector, config, threshold=threshold)


def run_detector_stream(
    stream: SimulatedRun,
    detector: object,
    config: ExperimentConfig,
    *,
    threshold: float | None = None,
) -> dict[str, object]:
    max_statistic = float("-inf")
    argmax_time: int | None = None
    argmax_channel: int | None = None
    argmax_arm: str | None = None
    alarm_time: int | None = None
    for idx, row in enumerate(stream.z):
        result = detector.update(row, endogenous_delta=stream.endogenous_delta[idx])
        for candidate in result.statistics:
            if int(candidate.time) <= int(config.warmup):
                continue
            statistic = float(candidate.statistic)
            if statistic > max_statistic:
                max_statistic = statistic
                argmax_time = int(candidate.time)
                argmax_channel = (
                    None if candidate.channel is None else int(candidate.channel)
                )
                argmax_arm = candidate.arm
            if threshold is not None and alarm_time is None and statistic > threshold:
                alarm_time = int(candidate.time)
    if not np.isfinite(max_statistic):
        max_statistic = float("nan")
    return {
        "max_statistic": max_statistic,
        "argmax_time": argmax_time,
        "argmax_channel": argmax_channel,
        "argmax_arm": argmax_arm,
        "alarm_time": alarm_time,
    }


def screened_baseline_factories(config: ExperimentConfig, settings: ScreenedReplaySettings):
    factories = default_detector_factories(config, include_references=False)
    return {
        f"{name}_screened": factories[name]
        for name in dict.fromkeys(settings.screened_baselines)
    }


def load_split_streams(
    settings: ScreenedReplaySettings,
    env_id: str,
    env_index: int,
    *,
    split: str,
    scenario: str,
) -> list[tuple[int, int, SimulatedRun]]:
    if split == "calibration":
        base_seed = settings.calibration_seed
        runs = settings.calibration_runs
    elif split == "heldout":
        base_seed = settings.heldout_seed
        runs = settings.heldout_runs
    elif split == "evaluation":
        base_seed = settings.evaluation_seed
        runs = settings.eval_runs
    else:
        raise ValueError(f"unknown split {split!r}")

    config = build_config(settings, base_seed, runs)
    records: list[tuple[int, int, SimulatedRun]] = []
    for run_idx in range(runs):
        seed = scenario_seed(base_seed, run_idx, scenario, env_index)
        stream = generate_experiment_stream(config, scenario, seed, env_id=env_id)
        records.append((run_idx, seed, stream))
    return records


def add_false_alarm_intervals(summary: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    enriched = summary.copy()
    counts: list[int] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for _, row in enriched.iterrows():
        runs = int(row["runs"])
        false_alarms = int(round(float(row["false_alarm_rate"]) * runs))
        lower, upper = clopper_pearson_interval(false_alarms, runs, confidence)
        counts.append(false_alarms)
        lowers.append(lower)
        uppers.append(upper)
    enriched["false_alarm_count"] = counts
    enriched["far_ci_lower"] = lowers
    enriched["far_ci_upper"] = uppers
    enriched["far_ci_confidence"] = float(confidence)
    return enriched


def run_for_env(
    settings: ScreenedReplaySettings,
    env_id: str,
    env_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration_records = load_split_streams(
        settings,
        env_id,
        env_index,
        split="calibration",
        scenario="no_shift",
    )
    if not calibration_records:
        raise ValueError(f"no calibration streams for {env_id}")

    feature_names = calibration_records[0][2].feature_names
    channel_rows: list[dict[str, object]] = []
    selected_channels: list[int] = []
    config = build_config(settings, settings.calibration_seed, settings.calibration_runs)

    for channel in range(len(feature_names)):
        feature_name = str(feature_names[channel])
        allowed_by_name = channel_name_allowed(
            feature_name,
            include_regex=settings.screen_include_regex,
            exclude_regex=settings.screen_exclude_regex,
        )
        maxima: list[float] = []
        for _, _, stream in calibration_records:
            result = run_stream(
                masked_stream(stream, [channel]),
                config,
                settings.arms,
                dense_cauchy=settings.dense_cauchy,
                dense_cauchy_fraction=settings.dense_cauchy_fraction,
            )
            maxima.append(float(result["max_statistic"]))
        calibration_max = float(np.nanmax(np.asarray(maxima, dtype=float)))
        selected = bool(allowed_by_name and calibration_max < settings.screen_max_statistic)
        if selected:
            selected_channels.append(channel)
        channel_rows.append(
            {
                "env_id": env_id,
                "channel": channel,
                "feature_name": feature_name,
                "allowed_by_name": allowed_by_name,
                "single_channel_calibration_max": calibration_max,
                "selected": selected,
                "screen_max_statistic": settings.screen_max_statistic,
                "screen_include_regex": "|".join(settings.screen_include_regex),
                "screen_exclude_regex": "|".join(settings.screen_exclude_regex),
                "arms": "+".join(settings.arms),
                "dense_cauchy": settings.dense_cauchy,
                "dense_cauchy_fraction": settings.dense_cauchy_fraction,
            }
        )

    if not selected_channels:
        raise ValueError(
            f"channel screen selected no channels for {env_id}; "
            "relax --screen-max-statistic or the feature-name regex filters"
        )

    baseline_factories = screened_baseline_factories(config, settings)
    detector_labels = ("CADET_screened", *tuple(baseline_factories))
    maxima_rows: list[dict[str, object]] = []
    calibration_maxima: dict[str, list[float]] = {label: [] for label in detector_labels}
    for split, scenario, records in (
        ("calibration", "no_shift", calibration_records),
        (
            "heldout",
            "no_shift",
            load_split_streams(settings, env_id, env_index, split="heldout", scenario="no_shift"),
        ),
    ):
        split_seed = settings.calibration_seed if split == "calibration" else settings.heldout_seed
        split_config = build_config(
            settings,
            split_seed,
            settings.calibration_runs if split == "calibration" else settings.heldout_runs,
        )
        for run_idx, seed, stream in records:
            masked = masked_stream(stream, selected_channels)
            detector_results: dict[str, dict[str, object]] = {
                "CADET_screened": run_stream(
                    masked,
                    split_config,
                    settings.arms,
                    dense_cauchy=settings.dense_cauchy,
                    dense_cauchy_fraction=settings.dense_cauchy_fraction,
                )
            }
            for label, factory in baseline_factories.items():
                detector_results[label] = run_detector_stream(
                    masked,
                    factory(masked.z.shape[1]),
                    split_config,
                )
            for detector_label, result in detector_results.items():
                max_statistic = float(result["max_statistic"])
                if split == "calibration":
                    calibration_maxima[detector_label].append(max_statistic)
                maxima_rows.append(
                    {
                        "split": split,
                        "env_id": env_id,
                        "scenario": scenario,
                        "run": run_idx,
                        "seed": seed,
                        "detector": detector_label,
                        "max_statistic": max_statistic,
                        "argmax_time": result["argmax_time"],
                        "argmax_channel": (
                            None
                            if result["argmax_channel"] is None
                            else selected_channels[int(result["argmax_channel"])]
                        ),
                        "argmax_arm": result["argmax_arm"],
                        "selected_channels": len(selected_channels),
                    }
                )

    threshold_records: list[dict[str, object]] = []
    heldout_records: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    for detector_label in detector_labels:
        threshold = empirical_run_threshold(
            calibration_maxima[detector_label],
            settings.target_far,
        )
        thresholds[detector_label] = float(threshold)
        heldout_values = [
            float(row["max_statistic"])
            for row in maxima_rows
            if row["split"] == "heldout" and row["detector"] == detector_label
        ]
        heldout_false_alarms = int(
            np.sum(np.asarray(heldout_values, dtype=float) > threshold)
        )
        heldout_runs = len(heldout_values)
        heldout_far = heldout_false_alarms / heldout_runs if heldout_runs else float("nan")
        lower, upper = clopper_pearson_interval(heldout_false_alarms, heldout_runs)
        status = "admissible" if heldout_runs and heldout_far <= settings.target_far else "diagnostic_only"
        threshold_records.append(
            {
                "detector": detector_label,
                "env_id": env_id,
                "empirical_threshold": threshold,
                "target_far": settings.target_far,
                "calibration_runs": len(calibration_maxima[detector_label]),
                "selected_channels": len(selected_channels),
                "screen_max_statistic": settings.screen_max_statistic,
                "arms": "+".join(settings.arms),
            }
        )
        heldout_records.append(
            {
                "detector": detector_label,
                "env_id": env_id,
                "empirical_threshold": threshold,
                "target_far": settings.target_far,
                "heldout_runs": heldout_runs,
                "heldout_false_alarms": heldout_false_alarms,
                "heldout_far": heldout_far,
                "far_ci_lower": lower,
                "far_ci_upper": upper,
                "confidence": 0.95,
                "selected_channels": len(selected_channels),
                "status": status,
                "reason": (
                    "held-out point FAR within target"
                    if status == "admissible"
                    else "held-out point FAR exceeds target"
                ),
            }
        )

    threshold_rows = pd.DataFrame(threshold_records)
    heldout_rows = pd.DataFrame(heldout_records)

    result_rows: list[dict[str, object]] = []
    eval_config = build_config(settings, settings.evaluation_seed, settings.eval_runs)
    for scenario in settings.scenarios:
        records = load_split_streams(
            settings,
            env_id,
            env_index,
            split="evaluation",
            scenario=scenario,
        )
        for run_idx, seed, stream in records:
            masked = masked_stream(stream, selected_channels)
            detector_results = {
                "CADET_screened": run_stream(
                    masked,
                    eval_config,
                    settings.arms,
                    dense_cauchy=settings.dense_cauchy,
                    dense_cauchy_fraction=settings.dense_cauchy_fraction,
                    threshold=thresholds["CADET_screened"],
                )
            }
            for label, factory in baseline_factories.items():
                detector_results[label] = run_detector_stream(
                    masked,
                    factory(masked.z.shape[1]),
                    eval_config,
                    threshold=thresholds[label],
                )
            for detector_label, result in detector_results.items():
                alarm_time = result["alarm_time"]
                tau = stream.tau
                has_change = tau is not None
                false_alarm = alarm_time is not None and (
                    not has_change or alarm_time < int(tau)
                )
                detected = alarm_time is not None and has_change and alarm_time >= int(tau)
                if has_change:
                    delay = (
                        (int(alarm_time) - int(tau))
                        if detected and alarm_time is not None
                        else (stream.z.shape[0] - int(tau))
                    )
                else:
                    delay = np.nan
                result_rows.append(
                    {
                        "env_id": env_id,
                        "scenario": scenario,
                        "run": run_idx,
                        "seed": seed,
                        "detector": detector_label,
                        "threshold_mode": "empirical_statistic",
                        "empirical_threshold": thresholds[detector_label],
                        "max_statistic": result["max_statistic"],
                        "tau": tau,
                        "alarm_time": alarm_time,
                        "has_change": has_change,
                        "false_alarm": false_alarm,
                        "detected": detected,
                        "delay": delay,
                        "run_length": stream.z.shape[0],
                        "censored": has_change and not detected,
                        "selected_channels": len(selected_channels),
                        "argmax_time": result["argmax_time"],
                        "argmax_channel": (
                            None
                            if result["argmax_channel"] is None
                            else selected_channels[int(result["argmax_channel"])]
                        ),
                        "argmax_arm": result["argmax_arm"],
                    }
                )

    return (
        pd.DataFrame(channel_rows),
        pd.DataFrame(maxima_rows),
        threshold_rows,
        heldout_rows,
        pd.DataFrame(result_rows),
    )


def run(settings: ScreenedReplaySettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_settings(settings)
    settings.outdir.mkdir(parents=True, exist_ok=True)

    channel_tables: list[pd.DataFrame] = []
    maxima_tables: list[pd.DataFrame] = []
    threshold_tables: list[pd.DataFrame] = []
    heldout_tables: list[pd.DataFrame] = []
    result_tables: list[pd.DataFrame] = []

    for env_index, env_id in enumerate(settings.mujoco_envs):
        channels, maxima, thresholds, heldout, raw_results = run_for_env(
            settings,
            env_id,
            env_index,
        )
        channel_tables.append(channels)
        maxima_tables.append(maxima)
        threshold_tables.append(thresholds)
        heldout_tables.append(heldout)
        result_tables.append(raw_results)

    channel_table = pd.concat(channel_tables, ignore_index=True)
    maxima_table = pd.concat(maxima_tables, ignore_index=True)
    threshold_table = pd.concat(threshold_tables, ignore_index=True)
    heldout_table = pd.concat(heldout_tables, ignore_index=True)
    raw_results = pd.concat(result_tables, ignore_index=True)
    summary = add_false_alarm_intervals(summarize_results(raw_results))

    channel_table.to_csv(settings.outdir / "screened_channels.csv", index=False)
    maxima_table.to_csv(settings.outdir / "screened_run_max_statistics.csv", index=False)
    threshold_table.to_csv(settings.outdir / "screened_empirical_thresholds.csv", index=False)
    heldout_table.to_csv(settings.outdir / "screened_heldout_far.csv", index=False)
    raw_results.to_csv(settings.outdir / "screened_raw_results.csv", index=False)
    summary.to_csv(settings.outdir / "screened_summary.csv", index=False)

    manifest = {
        "mujoco_envs": list(settings.mujoco_envs),
        "scenarios": list(settings.scenarios),
        "target_far": settings.target_far,
        "horizon": settings.horizon,
        "tau": settings.tau,
        "calibration_runs": settings.calibration_runs,
        "heldout_runs": settings.heldout_runs,
        "eval_runs": settings.eval_runs,
        "scale_window": settings.scale_window,
        "dynamics_window": settings.dynamics_window,
        "warmup": settings.warmup,
        "mujoco_feature_mode": settings.mujoco_feature_mode,
        "mujoco_policy_std": settings.mujoco_policy_std,
        "mujoco_policy_mode": settings.mujoco_policy_mode,
        "mujoco_probe_update_mode": settings.mujoco_probe_update_mode,
        "mujoco_frozen_buffer_size": settings.mujoco_frozen_buffer_size,
        "gravity_scale": settings.gravity_scale,
        "friction_scale": settings.friction_scale,
        "mujoco_sensor_bias_scale": settings.sensor_bias_scale,
        "mujoco_actuator_loss_scale": settings.actuator_loss_scale,
        "arms": list(settings.arms),
        "dense_cauchy": settings.dense_cauchy,
        "dense_cauchy_fraction": settings.dense_cauchy_fraction,
        "screened_baselines": list(settings.screened_baselines),
        "screen_max_statistic": settings.screen_max_statistic,
        "screen_include_regex": list(settings.screen_include_regex),
        "screen_exclude_regex": list(settings.screen_exclude_regex),
        "calibration_seed": settings.calibration_seed,
        "heldout_seed": settings.heldout_seed,
        "evaluation_seed": settings.evaluation_seed,
        "load_traces": None if settings.load_traces is None else str(settings.load_traces),
        "save_traces": None if settings.save_traces is None else str(settings.save_traces),
    }
    (settings.outdir / "screened_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return raw_results, summary


def main(argv: Sequence[str] | None = None) -> None:
    settings = settings_from_args(build_parser().parse_args(argv))
    _, summary = run(settings)
    print(summary.to_string(index=False))
    print(f"\nWrote screened replay artifacts to {settings.outdir}")


if __name__ == "__main__":
    main()
