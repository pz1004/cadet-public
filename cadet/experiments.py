"""Command-line entry point for CADET experiments."""

from __future__ import annotations

import argparse

from cadet.evaluation import ExperimentConfig, run_experiment_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CADET change-detection experiments.")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke experiment.")
    parser.add_argument("--runs", type=int, default=None, help="Number of Monte Carlo runs.")
    parser.add_argument("--horizon", type=int, default=None, help="Steps per run.")
    parser.add_argument("--tau", type=int, default=None, help="Ground-truth change point.")
    parser.add_argument("--alpha", type=float, default=0.05, help="False-alarm budget.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--source",
        choices=["synthetic", "mujoco"],
        default="synthetic",
        help="Benchmark source.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=[
            "no_shift",
            "scale",
            "dynamics",
            "mixed",
            "mean",
            "gravity",
            "friction",
            "target",
            "sensor_bias",
            "actuator_loss",
        ],
        help="Scenarios to run.",
    )
    parser.add_argument(
        "--mujoco-env",
        default="HalfCheetah-v5",
        help="Single Gymnasium MuJoCo environment id; kept for backward compatibility.",
    )
    parser.add_argument(
        "--mujoco-envs",
        nargs="+",
        default=None,
        help="Gymnasium MuJoCo environment ids, e.g. HalfCheetah-v5 Walker2d-v5.",
    )
    parser.add_argument("--gravity-scale", type=float, default=1.45, help="MuJoCo gravity multiplier.")
    parser.add_argument("--friction-scale", type=float, default=1.8, help="MuJoCo friction multiplier.")
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
        help="Action source for MuJoCo traces; non-default modes are diagnostics.",
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
    parser.add_argument("--save-traces", default=None, help="Directory for persisted generated traces.")
    parser.add_argument("--load-traces", default=None, help="Directory of persisted traces to reuse.")
    parser.add_argument("--calibration-runs", type=int, default=20, help="No-shift runs for matched-FAR calibration.")
    parser.add_argument("--eval-runs", type=int, default=None, help="Evaluation runs per env/scenario.")
    parser.add_argument(
        "--center-mode",
        choices=["reference", "block"],
        default="reference",
        help="Scale-arm center mode; 'reference' is paper-faithful.",
    )
    parser.add_argument(
        "--lambda-phi-mode",
        choices=["fixed", "warmup"],
        default="fixed",
        help="Use fixed or warm-up-estimated conservative mixing inflation.",
    )
    parser.add_argument(
        "--reference-center-warmup",
        type=int,
        default=None,
        help="Optional number of initial values used to freeze the reference center.",
    )
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument(
        "--per-step",
        action="store_true",
        help="Use per-step alpha spending instead of the paper's anytime schedule.",
    )
    parser.add_argument(
        "--anytime",
        action="store_true",
        help="Explicitly use the strict infinite-horizon anytime alpha-spending schedule.",
    )
    parser.add_argument(
        "--match-far",
        action="store_true",
        help="Calibrate detector alphas on no-shift runs before evaluation.",
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="Include robust scale/dynamics GLR reference procedures.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quick:
        runs = args.runs or 3
        horizon = args.horizon or 600
        tau = args.tau or 300
        outdir = args.outdir or "results/quick"
        if args.source == "mujoco":
            scenarios = args.scenarios or ["no_shift", "gravity"]
        else:
            scenarios = args.scenarios or ["no_shift", "scale", "dynamics", "mixed"]
        include_references = args.include_references
    else:
        if args.source == "mujoco":
            runs = args.runs or args.eval_runs or 20
            horizon = args.horizon or 2000
            tau = args.tau or 1000
        else:
            runs = args.runs or args.eval_runs or 50
            horizon = args.horizon or 1200
            tau = args.tau or 600
        outdir = args.outdir or "results/full"
        scenarios = args.scenarios or (
            ["no_shift", "gravity", "friction"] if args.source == "mujoco" else None
        )
        include_references = True if not args.include_references else args.include_references

    if args.source == "mujoco":
        if args.mujoco_envs is not None:
            mujoco_envs = tuple(args.mujoco_envs)
        elif args.mujoco_env != "HalfCheetah-v5":
            mujoco_envs = (args.mujoco_env,)
        else:
            mujoco_envs = ("HalfCheetah-v5", "Walker2d-v5")
    else:
        mujoco_envs = ()

    save_traces = args.save_traces
    if args.source == "mujoco" and save_traces is None and args.load_traces is None:
        save_traces = f"{outdir}/traces"

    config = ExperimentConfig(
        alpha=args.alpha,
        horizon=horizon,
        tau=tau,
        n_runs=runs,
        seed=args.seed,
        anytime=not args.per_step,
        source=args.source,
        mujoco_env_id=args.mujoco_env,
        mujoco_env_ids=mujoco_envs,
        mujoco_gravity_scale=args.gravity_scale,
        mujoco_friction_scale=args.friction_scale,
        mujoco_target_shift=args.mujoco_target_shift,
        mujoco_sensor_bias_scale=args.mujoco_sensor_bias_scale,
        mujoco_actuator_loss_scale=args.mujoco_actuator_loss_scale,
        mujoco_freeze_task_goal=args.mujoco_freeze_task_goal,
        mujoco_policy_std=args.mujoco_policy_std,
        mujoco_policy_mode=args.mujoco_policy_mode,
        mujoco_feature_mode=args.mujoco_feature_mode,
        mujoco_probe_update_mode=args.mujoco_probe_update_mode,
        mujoco_frozen_buffer_size=args.mujoco_frozen_buffer_size,
        mujoco_knn_neighbors=args.mujoco_knn_neighbors,
        save_traces=save_traces,
        load_traces=args.load_traces,
        calibration_runs=args.calibration_runs,
        eval_runs=args.eval_runs,
        center_mode=args.center_mode,
        lambda_phi_mode=args.lambda_phi_mode,
        reference_center_warmup=args.reference_center_warmup,
    )
    _, summary = run_experiment_suite(
        config,
        outdir=outdir,
        scenarios=scenarios,
        match_far=args.match_far,
        include_references=include_references,
    )
    print(summary.sort_values(["scenario", "detector"]).to_string(index=False))
    print(f"\nWrote results to {outdir}")


if __name__ == "__main__":
    main()
