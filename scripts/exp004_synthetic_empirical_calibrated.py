#!/usr/bin/env python
"""Experiment #4: synthetic validation with empirical baseline thresholds."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadet.calibration import (  # noqa: E402
    clopper_pearson_interval,
    evaluate_empirical_threshold_suite,
)
from cadet.evaluation import (  # noqa: E402
    ExperimentConfig,
    calibrate_alphas_to_far,
    default_detector_factories,
    detectability_rows_for_stream,
    evaluate_suite,
    generate_experiment_stream,
    summarize_results,
    write_summary_markdown,
)


DEFAULT_OUTDIR = ROOT / "results" / "exp004_synthetic_empirical_calibrated"
DEFAULT_THRESHOLDS = (
    ROOT / "results" / "exp003_empirical_baseline_calibration" / "empirical_thresholds.csv"
)
DEFAULT_STATUS = ROOT / "results" / "exp003_empirical_baseline_calibration" / "detector_status.csv"
DEFAULT_SCENARIOS = ("no_shift", "scale", "dynamics", "mixed", "mean")
DEFAULT_BASELINES = ("ADWIN", "BOCPD", "CUSUM", "Energy", "MMD", "Moment", "PageHinkley")
DEFAULT_CADET_DETECTORS = ("CADET", "CADET_no_PUR", "CADET_scale", "CADET_dynamics")
CALIBRATION_SEED = 20260619
EVALUATION_SEED = 20270619


@dataclass(frozen=True)
class Experiment4Settings:
    """Configuration for synthetic validation with repaired baseline calibration."""

    outdir: Path = DEFAULT_OUTDIR
    thresholds_path: Path = DEFAULT_THRESHOLDS
    detector_status_path: Path = DEFAULT_STATUS
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    cadet_detectors: tuple[str, ...] = DEFAULT_CADET_DETECTORS
    target_far: float = 0.10
    horizon: int = 650
    tau: int = 300
    calibration_runs: int = 20
    eval_runs: int = 20
    d: int = 5
    scale_window: int = 60
    dynamics_window: int = 80
    warmup: int = 120
    calibration_seed: int = CALIBRATION_SEED
    evaluation_seed: int = EVALUATION_SEED


def build_parser() -> argparse.ArgumentParser:
    """Build the Experiment #4 command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run synthetic validation using empirically calibrated baselines."
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--detector-status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--horizon", type=int, default=650)
    parser.add_argument("--tau", type=int, default=300)
    parser.add_argument("--target-far", type=float, default=0.10)
    parser.add_argument("--calibration-runs", type=int, default=20)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=DEFAULT_BASELINES,
        default=list(DEFAULT_BASELINES),
    )
    parser.add_argument(
        "--cadet-detectors",
        nargs="+",
        choices=DEFAULT_CADET_DETECTORS,
        default=list(DEFAULT_CADET_DETECTORS),
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Experiment4Settings:
    """Convert parsed command-line arguments into validated settings."""

    settings = Experiment4Settings(
        outdir=Path(args.outdir),
        thresholds_path=Path(args.thresholds),
        detector_status_path=Path(args.detector_status),
        scenarios=tuple(args.scenarios),
        baselines=tuple(args.baselines),
        cadet_detectors=tuple(args.cadet_detectors),
        target_far=float(args.target_far),
        horizon=int(args.horizon),
        tau=int(args.tau),
        calibration_runs=int(args.calibration_runs),
        eval_runs=int(args.eval_runs),
        calibration_seed=int(args.calibration_seed),
        evaluation_seed=int(args.evaluation_seed),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Experiment4Settings) -> None:
    """Validate settings and required Phase 1 artifacts."""

    if not 0.0 <= settings.target_far <= 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {settings.target_far}")
    if settings.horizon <= 1:
        raise ValueError(f"horizon must be greater than 1, got {settings.horizon}")
    if not 0 < settings.tau < settings.horizon:
        raise ValueError(f"tau must satisfy 0 < tau < horizon, got tau={settings.tau}")
    if settings.horizon < settings.warmup + settings.scale_window:
        raise ValueError(
            "horizon must be at least warmup + scale_window "
            f"({settings.warmup + settings.scale_window})"
        )
    if settings.calibration_runs <= 0 or settings.eval_runs <= 0:
        raise ValueError("calibration_runs and eval_runs must be positive")
    if not settings.thresholds_path.exists():
        raise ValueError(f"missing empirical thresholds: {settings.thresholds_path}")
    if not settings.detector_status_path.exists():
        raise ValueError(f"missing detector status table: {settings.detector_status_path}")


def build_config(seed: int, runs: int, settings: Experiment4Settings) -> ExperimentConfig:
    """Build a synthetic ExperimentConfig for one split."""

    return ExperimentConfig(
        alpha=settings.target_far,
        horizon=settings.horizon,
        tau=settings.tau,
        n_runs=runs,
        eval_runs=runs,
        calibration_runs=runs,
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


def _add_far_intervals(summary: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Add exact binomial intervals for false-alarm rates."""

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


def _detectability_audit(
    config: ExperimentConfig,
    scenarios: Sequence[str],
) -> pd.DataFrame:
    """Build the diagnostic residual-metric separation audit table."""

    rows: list[dict[str, object]] = []
    scenario_offsets = {
        "no_shift": 0,
        "scale": 1,
        "dynamics": 2,
        "mixed": 3,
        "mean": 4,
        "dense": 10,
        "dense_mean": 11,
    }
    n_runs = int(config.eval_runs if config.eval_runs is not None else config.n_runs)
    for scenario in scenarios:
        for run_idx in range(n_runs):
            seed = int(config.seed + 1009 * run_idx + 9173 * scenario_offsets.get(scenario, 99))
            stream = generate_experiment_stream(config, scenario, seed, env_id="synthetic")
            rows.extend(
                detectability_rows_for_stream(
                    stream,
                    scenario=scenario,
                    run_idx=run_idx,
                    seed=seed,
                    env_id="synthetic",
                    window=config.scale_window,
                )
            )
    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit["audit_status"] = "diagnostic_not_null_calibrated"
    return audit


def run(settings: Experiment4Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the synthetic empirical-threshold validation suite."""

    validate_settings(settings)
    settings.outdir.mkdir(parents=True, exist_ok=True)
    thresholds = pd.read_csv(settings.thresholds_path)
    detector_status = pd.read_csv(settings.detector_status_path)
    admissible = set(
        detector_status.loc[detector_status["status"] == "admissible", "detector"].astype(str)
    )
    baselines = tuple(name for name in settings.baselines if name in admissible)
    if not baselines:
        raise ValueError("no admissible baselines available from detector status table")

    calibration_config = build_config(
        settings.calibration_seed,
        settings.calibration_runs,
        settings,
    )
    evaluation_config = build_config(settings.evaluation_seed, settings.eval_runs, settings)
    cadet_alphas = calibrate_alphas_to_far(
        calibration_config,
        target_far=settings.target_far,
        n_calibration_runs=settings.calibration_runs,
        detector_names=list(settings.cadet_detectors),
        include_references=False,
    )
    pd.DataFrame(
        [
            {
                "detector": detector,
                "calibrated_alpha": alpha,
                "target_far": settings.target_far,
                "calibration_runs": settings.calibration_runs,
            }
            for detector, alpha in sorted(cadet_alphas.items())
        ]
    ).to_csv(settings.outdir / "cadet_calibrated_alphas.csv", index=False)

    factories = default_detector_factories(
        evaluation_config,
        cadet_alphas,
        include_references=False,
    )
    cadet_factories = {name: factories[name] for name in settings.cadet_detectors}
    cadet_results = evaluate_suite(
        evaluation_config,
        scenarios=list(settings.scenarios),
        detector_factories=cadet_factories,
        include_references=False,
    )
    cadet_results["threshold_mode"] = "matched_alpha"

    baseline_results = evaluate_empirical_threshold_suite(
        evaluation_config,
        settings.scenarios,
        thresholds[thresholds["detector"].isin(baselines)],
        detector_names=baselines,
    )
    raw_results = pd.concat([cadet_results, baseline_results], ignore_index=True)
    summary = _add_far_intervals(summarize_results(raw_results))
    ablations = summary[
        summary["detector"].isin(
            ["CADET", "CADET_no_PUR", "CADET_scale", "CADET_dynamics"]
        )
    ]
    audit = _detectability_audit(evaluation_config, settings.scenarios)

    raw_results.to_csv(settings.outdir / "raw_results.csv", index=False)
    summary.to_csv(settings.outdir / "summary.csv", index=False)
    ablations.to_csv(settings.outdir / "ablation_summary.csv", index=False)
    audit.to_csv(settings.outdir / "detectability_audit.csv", index=False)
    detector_status.to_csv(settings.outdir / "detector_status.csv", index=False)
    thresholds.to_csv(settings.outdir / "empirical_thresholds.csv", index=False)
    write_summary_markdown(summary, settings.outdir / "summary.md")
    manifest = {
        "source": "synthetic",
        "scenarios": list(settings.scenarios),
        "target_far": settings.target_far,
        "calibration_seed": settings.calibration_seed,
        "evaluation_seed": settings.evaluation_seed,
        "calibration_runs": settings.calibration_runs,
        "eval_runs": settings.eval_runs,
        "thresholds_path": str(settings.thresholds_path),
        "detector_status_path": str(settings.detector_status_path),
        "admissible_baselines": list(baselines),
        "cadet_detectors": list(settings.cadet_detectors),
    }
    (settings.outdir / "trace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return raw_results, summary


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for Experiment #4."""

    parser = build_parser()
    try:
        settings = settings_from_args(parser.parse_args(argv))
        _, summary = run(settings)
    except ValueError as exc:
        parser.error(str(exc))

    print(summary.sort_values(["scenario", "detector"]).to_string(index=False))
    print(f"\nWrote experiment #4 artifacts to {settings.outdir}")


if __name__ == "__main__":
    main()
