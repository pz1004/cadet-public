#!/usr/bin/env python
"""Experiment #2: diagnose and recalibrate synthetic baseline false alarms."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadet.evaluation import (  # noqa: E402
    ExperimentConfig,
    default_detector_factories,
    evaluate_suite,
    generate_experiment_stream,
    run_detector,
    summarize_results,
    write_summary_markdown,
)
from cadet.simulation import SimulatedRun  # noqa: E402


DEFAULT_OUTDIR = ROOT / "results" / "exp002_synthetic_baseline_calibration"
DEFAULT_SCENARIOS = ("no_shift", "scale", "dynamics", "mixed", "mean")
DEFAULT_BASELINES = ("ADWIN", "BOCPD", "CUSUM", "Energy", "MMD", "Moment", "PageHinkley")
DEFAULT_ALPHA_GRID = (
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    5e-2,
    1e-1,
    2e-1,
    5e-1,
)
CALIBRATION_SEED = 20260617
EVALUATION_SEED = 20270617
SYNTHETIC_ENV_ID = "synthetic"

DetectorFactory = Callable[[int], object]


@dataclass(frozen=True)
class Experiment2Settings:
    """Configuration for the held-out synthetic baseline calibration diagnostic."""

    outdir: Path = DEFAULT_OUTDIR
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    calibration_seed: int = CALIBRATION_SEED
    evaluation_seed: int = EVALUATION_SEED
    target_far: float = 0.10
    horizon: int = 650
    tau: int = 300
    runs: int = 6
    d: int = 5
    scale_window: int = 60
    dynamics_window: int = 80
    warmup: int = 120


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Experiment #2."""

    parser = argparse.ArgumentParser(
        description=(
            "Run held-out synthetic calibration diagnostics for baseline change detectors."
        )
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--runs", type=int, default=6, help="Calibration/evaluation runs.")
    parser.add_argument("--horizon", type=int, default=650, help="Steps per synthetic run.")
    parser.add_argument("--tau", type=int, default=300, help="Ground-truth change point.")
    parser.add_argument("--target-far", type=float, default=0.10, help="Target no-shift FAR.")
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
        help="Evaluation scenarios.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=DEFAULT_BASELINES,
        default=list(DEFAULT_BASELINES),
        help="Baseline detectors to calibrate and evaluate.",
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHA_GRID),
        help="Candidate alpha values; the largest alpha meeting target FAR is selected.",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Experiment2Settings:
    """Convert parsed command-line arguments into validated settings.

    Args:
        args: Parsed command-line namespace.

    Returns:
        A validated immutable settings object.

    Raises:
        ValueError: If any numeric or collection argument is invalid.
    """

    settings = Experiment2Settings(
        outdir=Path(args.outdir),
        scenarios=tuple(args.scenarios),
        baselines=tuple(args.baselines),
        alpha_grid=tuple(args.alpha_grid),
        calibration_seed=int(args.calibration_seed),
        evaluation_seed=int(args.evaluation_seed),
        target_far=float(args.target_far),
        horizon=int(args.horizon),
        tau=int(args.tau),
        runs=int(args.runs),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Experiment2Settings) -> None:
    """Validate experiment settings before running expensive work.

    Args:
        settings: Settings to validate.

    Raises:
        ValueError: If the settings are inconsistent or unsupported.
    """

    if not 0.0 <= settings.target_far <= 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {settings.target_far}")
    if settings.runs <= 0:
        raise ValueError(f"runs must be positive, got {settings.runs}")
    if settings.horizon <= 1:
        raise ValueError(f"horizon must be greater than 1, got {settings.horizon}")
    if not 0 < settings.tau < settings.horizon:
        raise ValueError(f"tau must satisfy 0 < tau < horizon, got tau={settings.tau}")
    if settings.warmup < 0:
        raise ValueError(f"warmup must be nonnegative, got {settings.warmup}")
    if settings.scale_window <= 0 or settings.dynamics_window <= 0:
        raise ValueError("scale_window and dynamics_window must be positive")
    if not settings.scenarios:
        raise ValueError("at least one scenario is required")
    if not settings.baselines:
        raise ValueError("at least one baseline detector is required")
    if not settings.alpha_grid:
        raise ValueError("alpha_grid must contain at least one value")
    invalid_alphas = [alpha for alpha in settings.alpha_grid if not 0.0 < alpha <= 1.0]
    if invalid_alphas:
        raise ValueError(f"alpha values must be in (0, 1], got {invalid_alphas}")


def build_config(seed: int, settings: Experiment2Settings) -> ExperimentConfig:
    """Create an ExperimentConfig for either calibration or held-out evaluation."""

    return ExperimentConfig(
        alpha=settings.target_far,
        horizon=settings.horizon,
        tau=settings.tau,
        n_runs=settings.runs,
        eval_runs=settings.runs,
        calibration_runs=settings.runs,
        d=settings.d,
        scale_window=settings.scale_window,
        dynamics_window=settings.dynamics_window,
        warmup=settings.warmup,
        anytime=False,
        seed=seed,
        source="synthetic",
        center_mode="reference",
        lambda_phi_mode="fixed",
    )


def _detector_factory(
    config: ExperimentConfig,
    detector_name: str,
    alpha: float,
) -> DetectorFactory:
    """Return one detector factory or fail with a helpful detector-name error."""

    factories = default_detector_factories(
        config,
        {detector_name: float(alpha)},
        include_references=False,
    )
    if detector_name not in factories:
        valid = ", ".join(sorted(factories))
        raise ValueError(f"unknown detector {detector_name!r}; valid detectors: {valid}")
    return factories[detector_name]


def _calibration_seeds(config: ExperimentConfig) -> tuple[int, ...]:
    """Return the no-shift calibration seeds used by evaluate_suite's seed schedule."""

    return tuple(int(config.seed + 1009 * run_idx) for run_idx in range(config.calibration_runs))


def _calibration_streams(config: ExperimentConfig) -> tuple[SimulatedRun, ...]:
    """Generate no-shift calibration streams once and replay them across alphas."""

    return tuple(
        generate_experiment_stream(
            config,
            "no_shift",
            seed,
            env_id=SYNTHETIC_ENV_ID,
        )
        for seed in _calibration_seeds(config)
    )


def _evaluate_no_shift_alpha(
    config: ExperimentConfig,
    detector_name: str,
    alpha: float,
    streams: Sequence[SimulatedRun],
    target_far: float,
) -> dict[str, Any]:
    """Evaluate one detector/alpha pair on cached no-shift streams.

    Args:
        config: Calibration experiment configuration.
        detector_name: Name understood by ``default_detector_factories``.
        alpha: Candidate detector alpha.
        streams: Cached no-shift calibration streams.
        target_far: Desired false-alarm rate.

    Returns:
        A row suitable for the calibration diagnostics CSV.
    """

    factory = _detector_factory(config, detector_name, alpha)
    false_alarms: list[bool] = []
    run_lengths: list[int] = []
    time_to_alarm: list[float] = []

    for stream in streams:
        alarm_time = run_detector(factory(stream.z.shape[1]), stream)
        run_length = int(stream.z.shape[0])
        false_alarm = alarm_time is not None
        false_alarms.append(false_alarm)
        run_lengths.append(run_length)
        time_to_alarm.append(float(alarm_time if alarm_time is not None else run_length))

    false_alarm_rate = float(np.mean(false_alarms)) if false_alarms else float("nan")
    return {
        "detector": detector_name,
        "alpha": float(alpha),
        "target_far": float(target_far),
        "calibration_runs": len(run_lengths),
        "false_alarm_rate": false_alarm_rate,
        "arl0_truncated": float(np.mean(time_to_alarm)) if time_to_alarm else float("nan"),
        "within_target": bool(false_alarm_rate <= target_far),
    }


def _select_calibrated_alpha(rows: Sequence[dict[str, Any]]) -> tuple[float, str, dict[str, Any]]:
    """Select the largest alpha within target FAR, or the safest failed alpha."""

    eligible = [row for row in rows if bool(row["within_target"])]
    if eligible:
        chosen = max(eligible, key=lambda row: float(row["alpha"]))
        return float(chosen["alpha"]), "ok", chosen

    chosen = min(
        rows,
        key=lambda row: (
            float(row["false_alarm_rate"]),
            float(row["alpha"]),
        ),
    )
    return float(chosen["alpha"]), "failed_target", chosen


def calibrate_baselines(
    config: ExperimentConfig,
    target_far: float,
    baselines: Sequence[str] = DEFAULT_BASELINES,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calibrate baseline alphas on held-out no-shift streams.

    Args:
        config: Synthetic no-shift calibration configuration.
        target_far: Maximum accepted calibration false-alarm rate.
        baselines: Detector names to calibrate.
        alpha_grid: Candidate alpha values.

    Returns:
        A mapping from detector name to selected alpha, a diagnostics table with all
        detector/alpha trials, and a compact selected-alpha table.
    """

    streams = _calibration_streams(config)
    diagnostics: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    calibrated: dict[str, float] = {}
    sorted_grid = tuple(sorted({float(alpha) for alpha in alpha_grid}))

    for detector_name in baselines:
        detector_rows = [
            _evaluate_no_shift_alpha(config, detector_name, alpha, streams, target_far)
            for alpha in sorted_grid
        ]
        diagnostics.extend(detector_rows)
        chosen_alpha, status, chosen = _select_calibrated_alpha(detector_rows)
        calibrated[detector_name] = chosen_alpha
        selected_rows.append(
            {
                "detector": detector_name,
                "calibrated_alpha": chosen_alpha,
                "target_far": float(target_far),
                "calibration_runs": int(chosen["calibration_runs"]),
                "observed_calibration_far": float(chosen["false_alarm_rate"]),
                "observed_calibration_arl0": float(chosen["arl0_truncated"]),
                "calibration_status": status,
            }
        )

    return calibrated, pd.DataFrame(diagnostics), pd.DataFrame(selected_rows)


def write_manifest(
    outdir: Path,
    settings: Experiment2Settings,
    evaluation_config: ExperimentConfig,
) -> None:
    """Write a manifest describing the held-out diagnostic run."""

    manifest = {
        "source": evaluation_config.source,
        "env_ids": [SYNTHETIC_ENV_ID],
        "save_traces": evaluation_config.save_traces,
        "load_traces": evaluation_config.load_traces,
        "calibration_seed": settings.calibration_seed,
        "evaluation_seed": settings.evaluation_seed,
        "scenarios": list(settings.scenarios),
        "baseline_detectors": list(settings.baselines),
        "alpha_grid": list(settings.alpha_grid),
    }
    (outdir / "trace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run(settings: Experiment2Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Experiment #2 and persist all artifacts.

    Args:
        settings: Validated experiment settings.

    Returns:
        The raw results, summary table, and selected calibration table.
    """

    validate_settings(settings)
    settings.outdir.mkdir(parents=True, exist_ok=True)
    calibration_config = build_config(settings.calibration_seed, settings)
    evaluation_config = build_config(settings.evaluation_seed, settings)

    calibrated_alphas, diagnostics, calibrated_table = calibrate_baselines(
        calibration_config,
        target_far=settings.target_far,
        baselines=settings.baselines,
        alpha_grid=settings.alpha_grid,
    )
    diagnostics.to_csv(settings.outdir / "baseline_calibration_diagnostics.csv", index=False)
    calibrated_table.to_csv(settings.outdir / "calibrated_alphas.csv", index=False)

    factories = default_detector_factories(
        evaluation_config,
        calibrated_alphas,
        include_references=False,
    )
    baseline_factories = {name: factories[name] for name in settings.baselines}
    results = evaluate_suite(
        evaluation_config,
        scenarios=list(settings.scenarios),
        detector_factories=baseline_factories,
        include_references=False,
    )
    summary = summarize_results(results)
    results.to_csv(settings.outdir / "raw_results.csv", index=False)
    summary.to_csv(settings.outdir / "summary.csv", index=False)
    write_summary_markdown(summary, settings.outdir / "summary.md")
    write_manifest(settings.outdir, settings, evaluation_config)
    return results, summary, calibrated_table


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for Experiment #2."""

    parser = build_parser()
    try:
        settings = settings_from_args(parser.parse_args(argv))
        previous_np_errors = np.seterr(all="ignore")
        try:
            _, summary, calibrated_table = run(settings)
        finally:
            np.seterr(**previous_np_errors)
    except ValueError as exc:
        parser.error(str(exc))

    print(calibrated_table.to_string(index=False))
    print()
    print(summary.sort_values(["scenario", "detector"]).to_string(index=False))
    print(f"\nWrote experiment #2 artifacts to {settings.outdir}")


if __name__ == "__main__":
    main()
