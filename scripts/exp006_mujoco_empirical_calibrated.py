#!/usr/bin/env python
"""Experiment #6: MuJoCo validation with empirical baseline thresholds."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import warnings

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadet.calibration import (  # noqa: E402
    clopper_pearson_interval,
    evaluate_empirical_threshold_suite,
    run_empirical_calibration,
    write_empirical_calibration_result,
)
from cadet.evaluation import (  # noqa: E402
    ExperimentConfig,
    calibrate_alphas_to_far,
    default_detector_factories,
    detectability_rows_for_stream,
    evaluate_suite,
    summarize_results,
    write_summary_markdown,
)


DEFAULT_OUTDIR = ROOT / "results" / "exp006_mujoco_empirical_calibrated"
DEFAULT_ENVS = ("HalfCheetah-v5", "Walker2d-v5")
DEFAULT_SCENARIOS = ("no_shift", "gravity", "friction")
MUJOCO_SCENARIOS = (
    "no_shift",
    "gravity",
    "friction",
    "target",
    "sensor_bias",
    "actuator_loss",
)
DEFAULT_BASELINES = (
    "ADWIN",
    "BOCPD",
    "CUSUM",
    "Energy",
    "MMD",
    "Moment",
    "PageHinkley",
)
REFERENCE_BASELINES = (
    "ScaleGLR",
    "DynamicsGLR",
)
ALL_BASELINES = DEFAULT_BASELINES + REFERENCE_BASELINES
DEFAULT_CADET_DETECTORS = ("CADET", "CADET_no_PUR", "CADET_scale", "CADET_dynamics")
ALL_CADET_DETECTORS = DEFAULT_CADET_DETECTORS + ("CADET_location",)
BASELINE_CALIBRATION_SEED = 20260620
BASELINE_HELDOUT_SEED = 20270620
CADET_CALIBRATION_SEED = 20280620
CADET_HELDOUT_SEED = 20285620
EVALUATION_SEED = 20290620


@dataclass(frozen=True)
class Experiment6Settings:
    """Configuration for MuJoCo validation with repaired baseline calibration."""

    outdir: Path = DEFAULT_OUTDIR
    mujoco_envs: tuple[str, ...] = DEFAULT_ENVS
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    baselines: tuple[str, ...] = DEFAULT_BASELINES
    cadet_detectors: tuple[str, ...] = DEFAULT_CADET_DETECTORS
    target_far: float = 0.05
    horizon: int = 2000
    tau: int = 1000
    baseline_calibration_runs: int = 20
    baseline_heldout_runs: int = 20
    cadet_calibration_runs: int = 20
    cadet_heldout_runs: int = 20
    eval_runs: int = 20
    scale_window: int = 80
    dynamics_window: int = 180
    warmup: int = 200
    gravity_scale: float = 1.45
    friction_scale: float = 1.8
    target_shift: float = 0.12
    sensor_bias_scale: float = 0.35
    actuator_loss_scale: float = 0.55
    freeze_task_goal: bool = False
    mujoco_policy_std: float = 0.35
    mujoco_policy_mode: str = "linear_actor"
    mujoco_feature_mode: str = "actor_value"
    mujoco_probe_update_mode: str = "online"
    mujoco_frozen_buffer_size: int = 96
    mujoco_knn_neighbors: int = 8
    center_mode: str = "reference"
    baseline_calibration_seed: int = BASELINE_CALIBRATION_SEED
    baseline_heldout_seed: int = BASELINE_HELDOUT_SEED
    cadet_calibration_seed: int = CADET_CALIBRATION_SEED
    cadet_heldout_seed: int = CADET_HELDOUT_SEED
    evaluation_seed: int = EVALUATION_SEED
    cadet_threshold_mode: str = "matched_alpha"
    calibration_scope: str = "pooled"
    baseline_thresholds: Path | None = None
    detector_status: Path | None = None
    cadet_alphas: Path | None = None
    cadet_thresholds: Path | None = None
    save_traces: Path | None = None
    load_traces: Path | None = None
    anytime: bool = True
    evaluation_only: bool = False
    allow_no_admissible_baselines: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build the Experiment #6 command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run MuJoCo validation with empirical baseline thresholds."
    )
    parser.add_argument("--quick", action="store_true", help="Run a tiny MuJoCo smoke variant.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--save-traces", type=Path, default=None)
    parser.add_argument("--load-traces", type=Path, default=None)
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Reuse saved calibration tables and only rerun detector evaluation.",
    )
    parser.add_argument("--baseline-thresholds", type=Path, default=None)
    parser.add_argument("--detector-status", type=Path, default=None)
    parser.add_argument("--cadet-alphas", type=Path, default=None)
    parser.add_argument("--mujoco-envs", nargs="+", default=None)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=MUJOCO_SCENARIOS,
        default=None,
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=ALL_BASELINES,
        default=list(DEFAULT_BASELINES),
    )
    parser.add_argument(
        "--include-glr-references",
        action="store_true",
        help="Include robust ScaleGLR/DynamicsGLR references; this is substantially slower.",
    )
    parser.add_argument(
        "--cadet-detectors",
        nargs="+",
        choices=ALL_CADET_DETECTORS,
        default=list(DEFAULT_CADET_DETECTORS),
    )
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--baseline-calibration-runs", type=int, default=None)
    parser.add_argument("--baseline-heldout-runs", type=int, default=None)
    parser.add_argument("--cadet-calibration-runs", type=int, default=None)
    parser.add_argument("--cadet-heldout-runs", type=int, default=None)
    parser.add_argument("--eval-runs", type=int, default=None)
    parser.add_argument("--scale-window", type=int, default=None)
    parser.add_argument("--dynamics-window", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--gravity-scale", type=float, default=1.45)
    parser.add_argument("--friction-scale", type=float, default=1.8)
    parser.add_argument(
        "--mujoco-target-shift",
        type=float,
        default=0.12,
        help="Goal/target position offset magnitude for Reacher/Pusher target-shift diagnostics.",
    )
    parser.add_argument(
        "--mujoco-sensor-bias-scale",
        type=float,
        default=0.35,
        help="Coordinatewise observation offset magnitude for the MuJoCo sensor-bias scenario.",
    )
    parser.add_argument(
        "--mujoco-actuator-loss-scale",
        type=float,
        default=0.55,
        help="Post-change multiplier applied to executed MuJoCo actions for actuator-loss diagnostics.",
    )
    parser.add_argument(
        "--mujoco-freeze-task-goal",
        action="store_true",
        help="Reapply the same MuJoCo task goal after episode resets for target-shift diagnostics.",
    )
    parser.add_argument(
        "--mujoco-policy-std",
        type=float,
        default=0.35,
        help="Gaussian action noise used by the lightweight MuJoCo probe policy.",
    )
    parser.add_argument(
        "--mujoco-policy-mode",
        choices=[
            "linear_actor",
            "linear_sinusoidal_dither",
            "zero",
            "sinusoidal",
            "chirp",
            "pulse",
        ],
        default="linear_actor",
        help=(
            "Action source for MuJoCo traces. The default linear_actor matches the paper's "
            "lightweight actor-value probe; linear_sinusoidal_dither adds a bounded active "
            "monitoring signal, while zero, sinusoidal, chirp, and pulse are explicit "
            "diagnostic policies."
        ),
    )
    parser.add_argument(
        "--mujoco-feature-mode",
        choices=[
            "actor_value",
            "actor_value_kinematic",
            "actor_value_telemetry",
            "actor_value_domain",
            "actor_value_reset_probe",
            "reset_probe",
            "model_probe",
            "model_state_probe",
            "model_vector_probe",
            "model_component_probe",
            "motion_probe",
            "locomotion_observation_probe",
            "locomotion_dynamics_probe",
            "locomotion_stability_probe",
            "gravity_signature_probe",
            "gravity_residual_probe",
            "gravity_compensation_probe",
            "gravity_balance_probe",
            "walker_gravity_response_probe",
            "walker_excitation_response_probe",
            "walker_innovation_gain_probe",
            "gait_phase_response_probe",
            "gait_load_response_probe",
            "mechanical_energy_probe",
            "control_response_probe",
            "actuator_efficiency_probe",
            "sensor_consistency_probe",
            "physics_residual_probe",
            "force_balance_probe",
            "contact_slip_probe",
            "support_telemetry_probe",
            "support_response_probe",
            "kernel_support_response_probe",
            "task_reward_probe",
            "task_observation_probe",
            "knn_state_probe",
            "phase_state_probe",
        ],
        default="actor_value",
        help=(
            "MuJoCo probe features; kinematic appends state/action norms, telemetry also "
            "appends simulator fields, domain omits absolute progress channels, and "
            "reset-probe evaluates a frozen state/action buffer in a resettable simulator."
        ),
    )
    parser.add_argument(
        "--mujoco-probe-update-mode",
        choices=["online", "frozen_policy"],
        default="online",
        help="Update the lightweight MuJoCo actor-value probe online or keep it fixed.",
    )
    parser.add_argument(
        "--mujoco-frozen-buffer-size",
        type=int,
        default=96,
        help="Frozen transition buffer size used for PUR and reset-probe features.",
    )
    parser.add_argument(
        "--mujoco-knn-neighbors",
        type=int,
        default=8,
        help="Number of neighbors used by the MuJoCo kNN state-transition probe.",
    )
    parser.add_argument(
        "--center-mode",
        choices=["reference", "block"],
        default="reference",
        help="Center scale-arm windows against the warmup reference or each detection block.",
    )
    parser.add_argument(
        "--baseline-calibration-seed",
        type=int,
        default=BASELINE_CALIBRATION_SEED,
    )
    parser.add_argument("--baseline-heldout-seed", type=int, default=BASELINE_HELDOUT_SEED)
    parser.add_argument(
        "--cadet-calibration-seed",
        type=int,
        default=CADET_CALIBRATION_SEED,
    )
    parser.add_argument("--cadet-heldout-seed", type=int, default=CADET_HELDOUT_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument(
        "--cadet-threshold-mode",
        choices=["matched_alpha", "empirical_statistic"],
        default="matched_alpha",
        help="Calibrate CADET by nominal alpha bisection or empirical run-level statistics.",
    )
    parser.add_argument(
        "--calibration-scope",
        choices=["pooled", "per_env"],
        default="pooled",
        help=(
            "Pool no-shift calibration streams across MuJoCo environments or calibrate "
            "separate empirical thresholds for each environment."
        ),
    )
    parser.add_argument("--cadet-thresholds", type=Path, default=None)
    parser.add_argument(
        "--per-step",
        action="store_true",
        help="Use per-step alpha spending instead of the anytime schedule.",
    )
    parser.add_argument(
        "--allow-no-admissible-baselines",
        action="store_true",
        help="Diagnostic mode: continue with CADET-only evaluation if no baseline passes FAR.",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Experiment6Settings:
    """Convert parsed arguments into validated experiment settings."""

    quick = bool(args.quick)
    outdir = Path(args.outdir)
    save_traces = args.save_traces
    load_traces = args.load_traces
    if save_traces is None and load_traces is None:
        save_traces = outdir / "traces"

    baselines = tuple(args.baselines)
    if args.include_glr_references:
        baselines = tuple(dict.fromkeys((*baselines, *REFERENCE_BASELINES)))

    settings = Experiment6Settings(
        outdir=outdir,
        mujoco_envs=tuple(args.mujoco_envs or (("HalfCheetah-v5",) if quick else DEFAULT_ENVS)),
        scenarios=tuple(args.scenarios or (("no_shift", "gravity") if quick else DEFAULT_SCENARIOS)),
        baselines=baselines,
        cadet_detectors=tuple(args.cadet_detectors),
        target_far=float(args.target_far),
        horizon=int(args.horizon or (80 if quick else 2000)),
        tau=int(args.tau or (40 if quick else 1000)),
        baseline_calibration_runs=int(args.baseline_calibration_runs or (1 if quick else 20)),
        baseline_heldout_runs=int(args.baseline_heldout_runs or (1 if quick else 20)),
        cadet_calibration_runs=int(args.cadet_calibration_runs or (1 if quick else 20)),
        cadet_heldout_runs=int(args.cadet_heldout_runs or (1 if quick else 20)),
        eval_runs=int(args.eval_runs or (1 if quick else 20)),
        scale_window=int(args.scale_window or (25 if quick else 80)),
        dynamics_window=int(args.dynamics_window or (50 if quick else 180)),
        warmup=int(args.warmup or (20 if quick else 200)),
        gravity_scale=float(args.gravity_scale),
        friction_scale=float(args.friction_scale),
        target_shift=float(args.mujoco_target_shift),
        sensor_bias_scale=float(args.mujoco_sensor_bias_scale),
        actuator_loss_scale=float(args.mujoco_actuator_loss_scale),
        freeze_task_goal=bool(args.mujoco_freeze_task_goal),
        mujoco_policy_std=float(args.mujoco_policy_std),
        mujoco_policy_mode=str(args.mujoco_policy_mode),
        mujoco_feature_mode=str(args.mujoco_feature_mode),
        mujoco_probe_update_mode=str(args.mujoco_probe_update_mode),
        mujoco_frozen_buffer_size=int(args.mujoco_frozen_buffer_size),
        mujoco_knn_neighbors=int(args.mujoco_knn_neighbors),
        center_mode=str(args.center_mode),
        baseline_calibration_seed=int(args.baseline_calibration_seed),
        baseline_heldout_seed=int(args.baseline_heldout_seed),
        cadet_calibration_seed=int(args.cadet_calibration_seed),
        cadet_heldout_seed=int(args.cadet_heldout_seed),
        evaluation_seed=int(args.evaluation_seed),
        cadet_threshold_mode=str(args.cadet_threshold_mode),
        calibration_scope=str(args.calibration_scope),
        baseline_thresholds=args.baseline_thresholds,
        detector_status=args.detector_status,
        cadet_alphas=args.cadet_alphas,
        cadet_thresholds=args.cadet_thresholds,
        save_traces=None if save_traces is None else Path(save_traces),
        load_traces=None if load_traces is None else Path(load_traces),
        anytime=not bool(args.per_step),
        evaluation_only=bool(args.evaluation_only),
        allow_no_admissible_baselines=bool(args.allow_no_admissible_baselines),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Experiment6Settings) -> None:
    """Validate Experiment #6 settings before running MuJoCo jobs."""

    if not settings.mujoco_envs:
        raise ValueError("at least one MuJoCo environment is required")
    if not settings.scenarios:
        raise ValueError("at least one scenario is required")
    if "no_shift" not in settings.scenarios:
        raise ValueError("scenarios must include no_shift for FAR reporting")
    unknown_scenarios = sorted(set(settings.scenarios) - set(MUJOCO_SCENARIOS))
    if unknown_scenarios:
        raise ValueError(f"unknown MuJoCo scenarios: {unknown_scenarios}")
    if not 0.0 <= settings.target_far <= 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {settings.target_far}")
    if settings.horizon <= 1:
        raise ValueError(f"horizon must be greater than 1, got {settings.horizon}")
    if not 0 < settings.tau < settings.horizon:
        raise ValueError(f"tau must satisfy 0 < tau < horizon, got tau={settings.tau}")
    if settings.mujoco_feature_mode not in {
        "actor_value",
        "actor_value_kinematic",
        "actor_value_telemetry",
        "actor_value_domain",
        "actor_value_reset_probe",
        "reset_probe",
        "model_probe",
        "model_state_probe",
        "model_vector_probe",
        "model_component_probe",
        "motion_probe",
        "locomotion_observation_probe",
        "locomotion_dynamics_probe",
        "locomotion_stability_probe",
        "gravity_signature_probe",
        "gravity_residual_probe",
        "gravity_compensation_probe",
        "gravity_balance_probe",
        "walker_gravity_response_probe",
        "walker_excitation_response_probe",
        "walker_innovation_gain_probe",
        "gait_phase_response_probe",
        "gait_load_response_probe",
        "mechanical_energy_probe",
        "control_response_probe",
        "actuator_efficiency_probe",
        "sensor_consistency_probe",
        "physics_residual_probe",
        "force_balance_probe",
        "contact_slip_probe",
        "support_telemetry_probe",
        "support_response_probe",
        "kernel_support_response_probe",
        "task_reward_probe",
        "task_observation_probe",
        "knn_state_probe",
        "phase_state_probe",
    }:
        raise ValueError(f"unknown MuJoCo feature mode: {settings.mujoco_feature_mode}")
    if settings.mujoco_probe_update_mode not in {"online", "frozen_policy"}:
        raise ValueError(
            f"unknown MuJoCo probe update mode: {settings.mujoco_probe_update_mode}"
        )
    if settings.mujoco_policy_std <= 0.0:
        raise ValueError(f"mujoco_policy_std must be positive, got {settings.mujoco_policy_std}")
    if settings.mujoco_policy_mode not in {
        "linear_actor",
        "linear_sinusoidal_dither",
        "linear_probe_burst",
        "zero",
        "sinusoidal",
        "chirp",
        "pulse",
    }:
        raise ValueError(f"unknown MuJoCo policy mode: {settings.mujoco_policy_mode}")
    if settings.target_shift <= 0.0:
        raise ValueError(f"mujoco_target_shift must be positive, got {settings.target_shift}")
    if settings.sensor_bias_scale <= 0.0:
        raise ValueError(
            f"mujoco_sensor_bias_scale must be positive, got {settings.sensor_bias_scale}"
        )
    if not 0.0 < settings.actuator_loss_scale <= 1.0:
        raise ValueError(
            "mujoco_actuator_loss_scale must be in (0, 1], "
            f"got {settings.actuator_loss_scale}"
        )
    if settings.mujoco_frozen_buffer_size <= 0:
        raise ValueError(
            f"mujoco_frozen_buffer_size must be positive, got {settings.mujoco_frozen_buffer_size}"
        )
    if settings.mujoco_knn_neighbors <= 0:
        raise ValueError(
            f"mujoco_knn_neighbors must be positive, got {settings.mujoco_knn_neighbors}"
        )
    if settings.center_mode not in {"reference", "block"}:
        raise ValueError(f"unknown CADET center mode: {settings.center_mode}")
    if settings.cadet_threshold_mode not in {"matched_alpha", "empirical_statistic"}:
        raise ValueError(f"unknown CADET threshold mode: {settings.cadet_threshold_mode}")
    if settings.calibration_scope not in {"pooled", "per_env"}:
        raise ValueError(f"unknown calibration scope: {settings.calibration_scope}")
    if (
        settings.calibration_scope == "per_env"
        and settings.cadet_threshold_mode == "matched_alpha"
    ):
        raise ValueError("per-env calibration currently requires empirical_statistic CADET mode")
    if settings.horizon < settings.warmup + settings.scale_window:
        raise ValueError(
            "horizon must be at least warmup + scale_window "
            f"({settings.warmup + settings.scale_window})"
        )
    run_counts = (
        settings.baseline_calibration_runs,
        settings.baseline_heldout_runs,
        settings.cadet_calibration_runs,
        settings.cadet_heldout_runs,
        settings.eval_runs,
    )
    if any(count <= 0 for count in run_counts):
        raise ValueError("all calibration, held-out, and evaluation run counts must be positive")
    if settings.load_traces is not None and not settings.load_traces.exists():
        raise ValueError(f"load-traces directory does not exist: {settings.load_traces}")
    if settings.evaluation_only:
        required = {
            "baseline-thresholds": settings.baseline_thresholds,
            "detector-status": settings.detector_status,
        }
        if settings.cadet_threshold_mode == "matched_alpha":
            required["cadet-alphas"] = settings.cadet_alphas
        elif settings.cadet_threshold_mode == "empirical_statistic":
            required["cadet-thresholds"] = settings.cadet_thresholds
        else:
            raise ValueError(f"unknown CADET threshold mode: {settings.cadet_threshold_mode}")
        missing = [name for name, path in required.items() if path is None or not path.exists()]
        if missing:
            raise ValueError(
                "evaluation-only mode requires existing "
                + ", ".join(sorted(missing))
                + " files"
            )


def build_config(seed: int, runs: int, settings: Experiment6Settings) -> ExperimentConfig:
    """Build a MuJoCo ExperimentConfig for one split."""

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
        anytime=settings.anytime,
        seed=seed,
        source="mujoco",
        mujoco_env_id=settings.mujoco_envs[0],
        mujoco_env_ids=settings.mujoco_envs,
        mujoco_gravity_scale=settings.gravity_scale,
        mujoco_friction_scale=settings.friction_scale,
        mujoco_target_shift=settings.target_shift,
        mujoco_sensor_bias_scale=settings.sensor_bias_scale,
        mujoco_actuator_loss_scale=settings.actuator_loss_scale,
        mujoco_freeze_task_goal=settings.freeze_task_goal,
        mujoco_policy_std=settings.mujoco_policy_std,
        mujoco_policy_mode=settings.mujoco_policy_mode,
        mujoco_feature_mode=settings.mujoco_feature_mode,
        mujoco_probe_update_mode=settings.mujoco_probe_update_mode,
        mujoco_frozen_buffer_size=settings.mujoco_frozen_buffer_size,
        mujoco_knn_neighbors=settings.mujoco_knn_neighbors,
        save_traces=None if settings.save_traces is None else str(settings.save_traces),
        load_traces=None if settings.load_traces is None else str(settings.load_traces),
        center_mode=settings.center_mode,
        lambda_phi_mode="fixed",
    )


def _add_far_intervals(summary: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Add exact binomial false-alarm intervals to a summary table."""

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


def _trace_manifest(settings: Experiment6Settings) -> dict[str, object]:
    trace_dir = settings.load_traces or settings.save_traces
    if trace_dir is None:
        return {"traces": []}
    manifest_path = Path(trace_dir) / "trace_manifest.json"
    if not manifest_path.exists():
        return {"trace_dir": str(trace_dir), "traces": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trace_dir"] = str(trace_dir)
    return manifest


def _write_run_manifest(
    settings: Experiment6Settings,
    admissible_baselines: Sequence[str],
) -> None:
    manifest = {
        "source": "mujoco",
        "mujoco_envs": list(settings.mujoco_envs),
        "scenarios": list(settings.scenarios),
        "target_far": settings.target_far,
        "horizon": settings.horizon,
        "tau": settings.tau,
        "mujoco_feature_mode": settings.mujoco_feature_mode,
        "mujoco_probe_update_mode": settings.mujoco_probe_update_mode,
        "mujoco_target_shift": settings.target_shift,
        "mujoco_sensor_bias_scale": settings.sensor_bias_scale,
        "mujoco_actuator_loss_scale": settings.actuator_loss_scale,
        "mujoco_freeze_task_goal": settings.freeze_task_goal,
        "mujoco_policy_std": settings.mujoco_policy_std,
        "mujoco_policy_mode": settings.mujoco_policy_mode,
        "mujoco_frozen_buffer_size": settings.mujoco_frozen_buffer_size,
        "mujoco_knn_neighbors": settings.mujoco_knn_neighbors,
        "center_mode": settings.center_mode,
        "baseline_calibration_runs": settings.baseline_calibration_runs,
        "baseline_heldout_runs": settings.baseline_heldout_runs,
        "cadet_calibration_runs": settings.cadet_calibration_runs,
        "cadet_heldout_runs": settings.cadet_heldout_runs,
        "eval_runs": settings.eval_runs,
        "baseline_calibration_seed": settings.baseline_calibration_seed,
        "baseline_heldout_seed": settings.baseline_heldout_seed,
        "cadet_calibration_seed": settings.cadet_calibration_seed,
        "cadet_heldout_seed": settings.cadet_heldout_seed,
        "evaluation_seed": settings.evaluation_seed,
        "cadet_threshold_mode": settings.cadet_threshold_mode,
        "calibration_scope": settings.calibration_scope,
        "save_traces": None if settings.save_traces is None else str(settings.save_traces),
        "load_traces": None if settings.load_traces is None else str(settings.load_traces),
        "anytime": settings.anytime,
        "evaluation_only": settings.evaluation_only,
        "allow_no_admissible_baselines": settings.allow_no_admissible_baselines,
        "baseline_thresholds": (
            None if settings.baseline_thresholds is None else str(settings.baseline_thresholds)
        ),
        "detector_status": None if settings.detector_status is None else str(settings.detector_status),
        "cadet_alphas": None if settings.cadet_alphas is None else str(settings.cadet_alphas),
        "cadet_thresholds": (
            None if settings.cadet_thresholds is None else str(settings.cadet_thresholds)
        ),
        "baselines_requested": list(settings.baselines),
        "admissible_baselines": list(admissible_baselines),
        "cadet_detectors": list(settings.cadet_detectors),
        "trace_manifest": _trace_manifest(settings),
    }
    (settings.outdir / "trace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _maybe_copy_table(source: Path | None, destination: Path) -> None:
    if source is not None and source.exists():
        pd.read_csv(source).to_csv(destination, index=False)


def _admissible_thresholds(
    threshold_table: pd.DataFrame,
    detector_status: pd.DataFrame,
) -> pd.DataFrame:
    """Filter thresholds to detector/env rows that passed held-out FAR."""

    admissible = detector_status[detector_status["status"] == "admissible"].copy()
    if threshold_table.empty or admissible.empty:
        return threshold_table.iloc[0:0].copy()
    if "env_id" in threshold_table.columns and "env_id" in admissible.columns:
        keys = admissible[["detector", "env_id"]].drop_duplicates()
        return threshold_table.merge(keys, on=["detector", "env_id"], how="inner")
    return threshold_table[
        threshold_table["detector"].isin(admissible["detector"].astype(str))
    ].copy()


def run(settings: Experiment6Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the MuJoCo empirical-threshold validation suite."""

    validate_settings(settings)
    settings.outdir.mkdir(parents=True, exist_ok=True)
    if settings.save_traces is not None:
        settings.save_traces.mkdir(parents=True, exist_ok=True)

    if settings.evaluation_only:
        assert settings.baseline_thresholds is not None
        assert settings.detector_status is not None
        baseline_threshold_table = pd.read_csv(settings.baseline_thresholds)
        detector_status = pd.read_csv(settings.detector_status)
        cadet_alpha_table = None
        cadet_threshold_table = None
        if settings.cadet_threshold_mode == "matched_alpha":
            assert settings.cadet_alphas is not None
            cadet_alpha_table = pd.read_csv(settings.cadet_alphas)
        else:
            assert settings.cadet_thresholds is not None
            cadet_threshold_table = pd.read_csv(settings.cadet_thresholds)
        _maybe_copy_table(settings.baseline_thresholds, settings.outdir / "empirical_thresholds.csv")
        _maybe_copy_table(settings.detector_status, settings.outdir / "detector_status.csv")
        if settings.cadet_threshold_mode == "matched_alpha":
            _maybe_copy_table(settings.cadet_alphas, settings.outdir / "calibrated_alphas.csv")
            _maybe_copy_table(settings.cadet_alphas, settings.outdir / "cadet_calibrated_alphas.csv")
        else:
            _maybe_copy_table(
                settings.cadet_thresholds,
                settings.outdir / "cadet_empirical_thresholds.csv",
            )
        source_dir = settings.detector_status.parent
        _maybe_copy_table(source_dir / "heldout_far.csv", settings.outdir / "heldout_far.csv")
        _maybe_copy_table(
            source_dir / "run_max_statistics.csv",
            settings.outdir / "run_max_statistics.csv",
        )
    else:
        baseline_calibration = build_config(
            settings.baseline_calibration_seed,
            settings.baseline_calibration_runs,
            settings,
        )
        baseline_heldout = build_config(
            settings.baseline_heldout_seed,
            settings.baseline_heldout_runs,
            settings,
        )
        baseline_result = run_empirical_calibration(
            baseline_calibration,
            baseline_heldout,
            settings.baselines,
            target_far=settings.target_far,
            group_by_env=settings.calibration_scope == "per_env",
        )
        write_empirical_calibration_result(baseline_result, settings.outdir)
        baseline_threshold_table = baseline_result.thresholds
        detector_status = baseline_result.detector_status

    baseline_thresholds = _admissible_thresholds(baseline_threshold_table, detector_status)
    admissible_baselines = tuple(dict.fromkeys(baseline_thresholds["detector"].astype(str)))
    if not admissible_baselines:
        if not settings.allow_no_admissible_baselines:
            raise ValueError("no admissible baselines after MuJoCo empirical calibration")
        warnings.warn(
            "No admissible baselines after MuJoCo empirical calibration; "
            "continuing with CADET-only diagnostic evaluation.",
            RuntimeWarning,
            stacklevel=2,
        )

    evaluation_config = build_config(settings.evaluation_seed, settings.eval_runs, settings)
    cadet_alphas: dict[str, float] = {}
    cadet_threshold_table: pd.DataFrame | None
    if settings.evaluation_only and settings.cadet_threshold_mode == "matched_alpha":
        assert cadet_alpha_table is not None
        cadet_alphas = dict(
            zip(
                cadet_alpha_table["detector"].astype(str),
                cadet_alpha_table["calibrated_alpha"].astype(float),
                strict=True,
            )
        )
    elif settings.evaluation_only:
        assert cadet_threshold_table is not None
    elif settings.cadet_threshold_mode == "matched_alpha":
        cadet_calibration = build_config(
            settings.cadet_calibration_seed,
            settings.cadet_calibration_runs,
            settings,
        )
        cadet_alphas = calibrate_alphas_to_far(
            cadet_calibration,
            target_far=settings.target_far,
            n_calibration_runs=settings.cadet_calibration_runs,
            detector_names=list(settings.cadet_detectors),
            include_references=False,
        )
        cadet_alpha_table = pd.DataFrame(
            [
                {
                    "detector": detector,
                    "calibrated_alpha": alpha,
                    "target_far": settings.target_far,
                    "calibration_runs": settings.cadet_calibration_runs,
                }
                for detector, alpha in sorted(cadet_alphas.items())
            ]
        )
        cadet_alpha_table.to_csv(settings.outdir / "calibrated_alphas.csv", index=False)
        cadet_alpha_table.to_csv(settings.outdir / "cadet_calibrated_alphas.csv", index=False)
    else:
        cadet_calibration = build_config(
            settings.cadet_calibration_seed,
            settings.cadet_calibration_runs,
            settings,
        )
        cadet_heldout = build_config(
            settings.cadet_heldout_seed,
            settings.cadet_heldout_runs,
            settings,
        )
        cadet_threshold_result = run_empirical_calibration(
            cadet_calibration,
            cadet_heldout,
            settings.cadet_detectors,
            target_far=settings.target_far,
            group_by_env=settings.calibration_scope == "per_env",
        )
        cadet_threshold_result.raw_statistics.to_csv(
            settings.outdir / "cadet_raw_statistic_log.csv",
            index=False,
        )
        cadet_threshold_result.run_maxima.to_csv(
            settings.outdir / "cadet_run_max_statistics.csv",
            index=False,
        )
        cadet_threshold_result.thresholds.to_csv(
            settings.outdir / "cadet_empirical_thresholds.csv",
            index=False,
        )
        cadet_threshold_result.heldout_far.to_csv(
            settings.outdir / "cadet_heldout_far.csv",
            index=False,
        )
        cadet_threshold_result.detector_status.to_csv(
            settings.outdir / "cadet_detector_status.csv",
            index=False,
        )
        cadet_threshold_table = cadet_threshold_result.thresholds

    audit_rows: list[dict[str, object]] = []

    def audit_callback(stream, scenario: str, run_idx: int, seed: int, env_id: str) -> None:
        audit_rows.extend(
            detectability_rows_for_stream(
                stream,
                scenario=scenario,
                run_idx=run_idx,
                seed=seed,
                env_id=env_id,
                window=evaluation_config.scale_window,
            )
        )

    if settings.cadet_threshold_mode == "matched_alpha":
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
            audit_rows=audit_rows,
        )
        cadet_results["threshold_mode"] = "matched_alpha"
    else:
        assert cadet_threshold_table is not None
        cadet_results = evaluate_empirical_threshold_suite(
            evaluation_config,
            settings.scenarios,
            cadet_threshold_table,
            detector_names=settings.cadet_detectors,
            audit_callback=audit_callback,
        )
        cadet_results["threshold_mode"] = "empirical_statistic"

    baseline_results = evaluate_empirical_threshold_suite(
        evaluation_config,
        settings.scenarios,
        baseline_thresholds,
        detector_names=admissible_baselines,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "The behavior of DataFrame concatenation with empty or all-NA "
                "entries is deprecated.*"
            ),
            category=FutureWarning,
        )
        raw_results = pd.concat([cadet_results, baseline_results], ignore_index=True)
    summary = _add_far_intervals(summarize_results(raw_results))
    ablations = summary[
        summary["detector"].isin(
            ["CADET", "CADET_no_PUR", "CADET_scale", "CADET_dynamics", "CADET_location"]
        )
    ]
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit["audit_status"] = "diagnostic_not_null_calibrated"

    raw_results.to_csv(settings.outdir / "raw_results.csv", index=False)
    summary.to_csv(settings.outdir / "summary.csv", index=False)
    ablations.to_csv(settings.outdir / "ablation_summary.csv", index=False)
    audit.to_csv(settings.outdir / "detectability_audit.csv", index=False)
    detector_status.to_csv(settings.outdir / "detector_status.csv", index=False)
    baseline_threshold_table.to_csv(settings.outdir / "empirical_thresholds.csv", index=False)
    write_summary_markdown(summary, settings.outdir / "summary.md")
    _write_run_manifest(settings, admissible_baselines)
    return raw_results, summary


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for Experiment #6."""

    parser = build_parser()
    try:
        settings = settings_from_args(parser.parse_args(argv))
        _, summary = run(settings)
    except ValueError as exc:
        parser.error(str(exc))

    print(summary.sort_values(["scenario", "env_id", "detector"]).to_string(index=False))
    print(f"\nWrote experiment #6 artifacts to {settings.outdir}")


if __name__ == "__main__":
    main()
