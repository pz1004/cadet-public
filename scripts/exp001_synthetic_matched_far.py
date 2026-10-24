#!/usr/bin/env python
"""Experiment #1: synthetic matched-FAR pilot for CADET and baselines."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadet.evaluation import ExperimentConfig, run_experiment_suite  # noqa: E402


DEFAULT_OUTDIR = ROOT / "results" / "exp001_synthetic_matched_far"
DEFAULT_SCENARIOS = ("no_shift", "scale", "dynamics", "mixed", "mean")


@dataclass(frozen=True)
class Experiment1Settings:
    """Configuration for the synthetic matched-FAR pilot."""

    outdir: Path = DEFAULT_OUTDIR
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    alpha: float = 0.10
    horizon: int = 650
    tau: int = 300
    runs: int = 6
    d: int = 5
    scale_window: int = 60
    dynamics_window: int = 80
    warmup: int = 120
    seed: int = 20260617


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Experiment #1."""

    parser = argparse.ArgumentParser(
        description="Run the synthetic matched-FAR CADET pilot experiment."
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--runs", type=int, default=6, help="Runs per scenario.")
    parser.add_argument("--horizon", type=int, default=650, help="Steps per synthetic run.")
    parser.add_argument("--tau", type=int, default=300, help="Ground-truth change point.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Target false-alarm rate.")
    parser.add_argument("--seed", type=int, default=20260617, help="Base random seed.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
        help="Synthetic scenarios to evaluate.",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Experiment1Settings:
    """Convert parsed CLI arguments into validated settings."""

    settings = Experiment1Settings(
        outdir=Path(args.outdir),
        scenarios=tuple(args.scenarios),
        alpha=float(args.alpha),
        horizon=int(args.horizon),
        tau=int(args.tau),
        runs=int(args.runs),
        seed=int(args.seed),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Experiment1Settings) -> None:
    """Validate Experiment #1 settings before running the benchmark."""

    if not 0.0 <= settings.alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {settings.alpha}")
    if settings.runs <= 0:
        raise ValueError(f"runs must be positive, got {settings.runs}")
    if settings.horizon <= 1:
        raise ValueError(f"horizon must be greater than 1, got {settings.horizon}")
    if not 0 < settings.tau < settings.horizon:
        raise ValueError(f"tau must satisfy 0 < tau < horizon, got tau={settings.tau}")
    if not settings.scenarios:
        raise ValueError("at least one scenario is required")


def build_config(settings: Experiment1Settings) -> ExperimentConfig:
    """Create the shared ExperimentConfig used by Experiment #1."""

    return ExperimentConfig(
        alpha=settings.alpha,
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
        seed=settings.seed,
        source="synthetic",
        center_mode="reference",
        lambda_phi_mode="fixed",
    )


def run(settings: Experiment1Settings) -> pd.DataFrame:
    """Run Experiment #1 and return the summary table."""

    validate_settings(settings)
    config = build_config(settings)
    _, summary = run_experiment_suite(
        config,
        outdir=settings.outdir,
        scenarios=list(settings.scenarios),
        match_far=True,
        target_far=config.alpha,
        include_references=False,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for Experiment #1."""

    parser = build_parser()
    try:
        settings = settings_from_args(parser.parse_args(argv))
        summary = run(settings)
    except ValueError as exc:
        parser.error(str(exc))

    print(summary.sort_values(["scenario", "detector"]).to_string(index=False))
    print(f"\nWrote experiment #1 artifacts to {settings.outdir}")


if __name__ == "__main__":
    main()
