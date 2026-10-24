#!/usr/bin/env python
"""Experiment #3: empirical statistic-threshold calibration for baselines."""

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
    run_empirical_calibration,
    write_empirical_calibration_result,
)
from cadet.evaluation import ExperimentConfig  # noqa: E402


DEFAULT_OUTDIR = ROOT / "results" / "exp003_empirical_baseline_calibration"
DEFAULT_BASELINES = ("ADWIN", "BOCPD", "CUSUM", "Energy", "MMD", "Moment", "PageHinkley")
CALIBRATION_SEED = 20260618
HELDOUT_SEED = 20270618


@dataclass(frozen=True)
class Experiment3Settings:
    """Configuration for empirical baseline threshold calibration."""

    outdir: Path = DEFAULT_OUTDIR
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    target_far: float = 0.10
    confidence: float = 0.95
    horizon: int = 650
    tau: int = 300
    calibration_runs: int = 20
    heldout_runs: int = 20
    d: int = 5
    scale_window: int = 60
    dynamics_window: int = 80
    warmup: int = 120
    calibration_seed: int = CALIBRATION_SEED
    heldout_seed: int = HELDOUT_SEED


def build_parser() -> argparse.ArgumentParser:
    """Build the Experiment #3 command-line parser."""

    parser = argparse.ArgumentParser(
        description="Empirically calibrate baseline detector statistics on no-shift streams."
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--horizon", type=int, default=650, help="Steps per synthetic run.")
    parser.add_argument("--tau", type=int, default=300, help="Ground-truth change point.")
    parser.add_argument("--target-far", type=float, default=0.10, help="Target run-level FAR.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Binomial CI confidence.")
    parser.add_argument("--calibration-runs", type=int, default=20)
    parser.add_argument("--heldout-runs", type=int, default=20)
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--heldout-seed", type=int, default=HELDOUT_SEED)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=DEFAULT_BASELINES,
        default=list(DEFAULT_BASELINES),
        help="Baseline detectors to calibrate.",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Experiment3Settings:
    """Convert parsed command-line arguments into validated settings."""

    settings = Experiment3Settings(
        outdir=Path(args.outdir),
        baselines=tuple(args.baselines),
        target_far=float(args.target_far),
        confidence=float(args.confidence),
        horizon=int(args.horizon),
        tau=int(args.tau),
        calibration_runs=int(args.calibration_runs),
        heldout_runs=int(args.heldout_runs),
        calibration_seed=int(args.calibration_seed),
        heldout_seed=int(args.heldout_seed),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Experiment3Settings) -> None:
    """Validate settings before running calibration."""

    if not 0.0 <= settings.target_far <= 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {settings.target_far}")
    if not 0.0 < settings.confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {settings.confidence}")
    if settings.horizon <= 1:
        raise ValueError(f"horizon must be greater than 1, got {settings.horizon}")
    if not 0 < settings.tau < settings.horizon:
        raise ValueError(f"tau must satisfy 0 < tau < horizon, got tau={settings.tau}")
    if settings.horizon < settings.warmup + settings.scale_window:
        raise ValueError(
            "horizon must be at least warmup + scale_window "
            f"({settings.warmup + settings.scale_window})"
        )
    if settings.calibration_runs <= 0 or settings.heldout_runs <= 0:
        raise ValueError("calibration_runs and heldout_runs must be positive")
    if not settings.baselines:
        raise ValueError("at least one baseline detector is required")


def build_config(seed: int, runs: int, settings: Experiment3Settings) -> ExperimentConfig:
    """Build the synthetic no-shift ExperimentConfig for one split."""

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


def write_manifest(settings: Experiment3Settings) -> None:
    """Write a manifest for the empirical calibration run."""

    manifest = {
        "source": "synthetic",
        "scenario": "no_shift",
        "baselines": list(settings.baselines),
        "target_far": settings.target_far,
        "confidence": settings.confidence,
        "horizon": settings.horizon,
        "tau": settings.tau,
        "calibration_runs": settings.calibration_runs,
        "heldout_runs": settings.heldout_runs,
        "calibration_seed": settings.calibration_seed,
        "heldout_seed": settings.heldout_seed,
    }
    (settings.outdir / "trace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_summary_markdown(settings: Experiment3Settings, detector_status: pd.DataFrame) -> None:
    """Write a compact markdown summary for the status ledger and paper audit."""

    lines = [
        "# Empirical Baseline Calibration Summary",
        "",
        f"Target FAR: `{settings.target_far}`",
        f"Calibration runs: `{settings.calibration_runs}`",
        f"Held-out runs: `{settings.heldout_runs}`",
        "",
        detector_status.to_markdown(index=False),
        "",
    ]
    (settings.outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(settings: Experiment3Settings):
    """Run empirical baseline calibration and persist artifacts."""

    validate_settings(settings)
    settings.outdir.mkdir(parents=True, exist_ok=True)
    calibration_config = build_config(
        settings.calibration_seed,
        settings.calibration_runs,
        settings,
    )
    heldout_config = build_config(settings.heldout_seed, settings.heldout_runs, settings)
    result = run_empirical_calibration(
        calibration_config,
        heldout_config,
        settings.baselines,
        target_far=settings.target_far,
        confidence=settings.confidence,
    )
    write_empirical_calibration_result(result, settings.outdir)
    write_summary_markdown(settings, result.detector_status)
    write_manifest(settings)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for Experiment #3."""

    parser = build_parser()
    try:
        settings = settings_from_args(parser.parse_args(argv))
        result = run(settings)
    except ValueError as exc:
        parser.error(str(exc))

    print(result.detector_status.to_string(index=False))
    print(f"\nWrote experiment #3 artifacts to {settings.outdir}")


if __name__ == "__main__":
    main()
