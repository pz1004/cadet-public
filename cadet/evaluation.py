"""Experiment orchestration for CADET and baseline detectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from cadet.baselines import (
    ADWINDetector,
    BaselineConfig,
    BOCPDGaussianDetector,
    CUSUMDetector,
    EnergyDistanceDetector,
    KernelMMDDetector,
    MomentMonitor,
    PageHinkleyDetector,
    ResidualizedDetector,
    RobustDynamicsGLRDetector,
    RobustScaleGLRDetector,
)
from cadet.detectors import CADETConfig, CADETDetector
from cadet.mujoco import MujocoTraceConfig, generate_mujoco_internal_stream
from cadet.residualization import IdentityResidualizer, LinearPURResidualizer
from cadet.simulation import SimulatedRun, generate_rl_internal_stream
from cadet.utils import dispersion_ks_distance, flip_rate


DetectorFactory = Callable[[int], object]


@dataclass
class ExperimentConfig:
    alpha: float = 0.05
    horizon: int = 1200
    tau: int = 600
    n_runs: int = 50
    d: int = 5
    scale_window: int = 80
    dynamics_window: int = 180
    warmup: int = 200
    anytime: bool = True
    seed: int = 12345
    source: str = "synthetic"
    mujoco_env_id: str = "HalfCheetah-v5"
    mujoco_env_ids: tuple[str, ...] = ()
    mujoco_gravity_scale: float = 1.45
    mujoco_friction_scale: float = 1.8
    mujoco_target_shift: float = 0.12
    mujoco_sensor_bias_scale: float = 0.35
    mujoco_actuator_loss_scale: float = 0.55
    mujoco_freeze_task_goal: bool = False
    mujoco_policy_lr: float = 2e-4
    mujoco_value_lr: float = 1e-3
    mujoco_policy_std: float = 0.35
    mujoco_policy_mode: str = "linear_actor"
    mujoco_feature_mode: str = "actor_value"
    mujoco_probe_update_mode: str = "online"
    mujoco_frozen_buffer_size: int = 96
    mujoco_knn_neighbors: int = 8
    save_traces: str | None = None
    load_traces: str | None = None
    calibration_runs: int = 20
    eval_runs: int | None = None
    center_mode: str = "reference"
    lambda_phi_mode: str = "fixed"
    reference_center_warmup: int | None = None


def default_detector_factories(
    config: ExperimentConfig,
    alpha_by_detector: dict[str, float] | None = None,
    include_references: bool = True,
) -> dict[str, DetectorFactory]:
    alpha_by_detector = alpha_by_detector or {}

    def alpha_for(name: str) -> float:
        return float(alpha_by_detector.get(name, config.alpha))

    cadet_config = dict(
        alpha=config.alpha,
        scale_window=config.scale_window,
        dynamics_window=config.dynamics_window,
        warmup=config.warmup,
        anytime=config.anytime,
        center_mode=config.center_mode,
        lambda_phi_mode=config.lambda_phi_mode,
        reference_center_warmup=config.reference_center_warmup,
    )
    base_config = BaselineConfig(
        alpha=config.alpha,
        window=config.scale_window,
        warmup=config.warmup,
        anytime=config.anytime,
    )
    glr_config = BaselineConfig(
        alpha=config.alpha,
        window=2 * config.scale_window,
        warmup=config.warmup,
        anytime=config.anytime,
    )
    factories: dict[str, DetectorFactory] = {
        "CADET": lambda d: CADETDetector(
            CADETConfig(**{**cadet_config, "alpha": alpha_for("CADET")}),
            residualizer=LinearPURResidualizer(),
        ),
        "CADET_no_PUR": lambda d: CADETDetector(
            CADETConfig(**{**cadet_config, "alpha": alpha_for("CADET_no_PUR")}),
            residualizer=IdentityResidualizer(),
        ),
        "CADET_scale": lambda d: CADETDetector(
            CADETConfig(**{**cadet_config, "alpha": alpha_for("CADET_scale")}, arms=("scale",)),
            residualizer=LinearPURResidualizer(),
        ),
        "CADET_dynamics": lambda d: CADETDetector(
            CADETConfig(
                **{**cadet_config, "alpha": alpha_for("CADET_dynamics")},
                arms=("dynamics",),
            ),
            residualizer=LinearPURResidualizer(),
        ),
        "CADET_location": lambda d: CADETDetector(
            CADETConfig(
                **{**cadet_config, "alpha": alpha_for("CADET_location")},
                arms=("location",),
            ),
            residualizer=LinearPURResidualizer(),
        ),
        "Moment": lambda d: MomentMonitor(
            BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for("Moment")})
        ),
        "CUSUM": lambda d: CUSUMDetector(
            BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for("CUSUM")})
        ),
        "ADWIN": lambda d: ADWINDetector(
            BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for("ADWIN")})
        ),
        "PageHinkley": lambda d: PageHinkleyDetector(
            BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for("PageHinkley")})
        ),
        "MMD": lambda d: KernelMMDDetector(
            BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for("MMD")})
        ),
        "Energy": lambda d: EnergyDistanceDetector(
            BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for("Energy")})
        ),
        "BOCPD": lambda d: BOCPDGaussianDetector(
            BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for("BOCPD")})
        ),
    }

    # PUR + baseline-statistic ablation (M5): identical residual stream to CADET,
    # so any remaining difference is attributable to the detection statistic and not
    # to residualization. Registered for every online baseline; the reference GLR
    # procedures are offline and are excluded.
    _pur_wrappable = {
        "Moment": MomentMonitor,
        "CUSUM": CUSUMDetector,
        "ADWIN": ADWINDetector,
        "PageHinkley": PageHinkleyDetector,
        "MMD": KernelMMDDetector,
        "Energy": EnergyDistanceDetector,
        "BOCPD": BOCPDGaussianDetector,
    }

    def _make_pur_factory(base_name: str, cls):
        pur_name = f"{base_name}_PUR"

        def factory(d: int):
            cfg = BaselineConfig(**{**base_config.__dict__, "alpha": alpha_for(pur_name)})
            return ResidualizedDetector(cls(cfg), LinearPURResidualizer(), name=pur_name)

        return factory

    for _base_name, _cls in _pur_wrappable.items():
        factories[f"{_base_name}_PUR"] = _make_pur_factory(_base_name, _cls)

    if include_references:
        factories.update(
            {
                "ScaleGLR": lambda d: RobustScaleGLRDetector(
                    BaselineConfig(**{**glr_config.__dict__, "alpha": alpha_for("ScaleGLR")})
                ),
                "DynamicsGLR": lambda d: RobustDynamicsGLRDetector(
                    BaselineConfig(**{**glr_config.__dict__, "alpha": alpha_for("DynamicsGLR")})
                ),
            }
        )
    return factories


def _mujoco_env_ids(config: ExperimentConfig) -> tuple[str, ...]:
    return tuple(config.mujoco_env_ids) if config.mujoco_env_ids else (config.mujoco_env_id,)


def _evaluation_env_ids(config: ExperimentConfig) -> tuple[str, ...]:
    if config.source == "mujoco":
        return _mujoco_env_ids(config)
    return ("synthetic",)


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _trace_filename(
    source: str,
    env_id: str,
    scenario: str,
    seed: int,
    trace_variant: str | None = None,
) -> str:
    variant = ""
    if trace_variant and trace_variant != "actor_value":
        variant = f"_{_safe_token(trace_variant)}"
    return (
        f"{_safe_token(source)}_{_safe_token(env_id)}_"
        f"{_safe_token(scenario)}{variant}_seed{int(seed)}.npz"
    )


def _update_trace_manifest(
    directory: Path,
    filename: str,
    *,
    source: str,
    env_id: str,
    scenario: str,
    seed: int,
    tau: int | None,
    horizon: int,
    feature_names: tuple[str, ...],
    trace_variant: str | None = None,
) -> None:
    manifest_path = directory / "trace_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"traces": []}
    entries = {
        entry.get("filename"): entry
        for entry in manifest.get("traces", [])
        if isinstance(entry, dict)
    }
    entries[filename] = {
        "filename": filename,
        "source": source,
        "env_id": env_id,
        "scenario": scenario,
        "seed": int(seed),
        "trace_variant": trace_variant,
        "tau": None if tau is None else int(tau),
        "horizon": int(horizon),
        "feature_names": list(feature_names),
    }
    manifest["traces"] = [entries[key] for key in sorted(entries)]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def save_trace(
    run: SimulatedRun,
    directory: str | Path,
    *,
    source: str,
    env_id: str,
    scenario: str,
    seed: int,
    trace_variant: str | None = None,
) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    filename = _trace_filename(source, env_id, scenario, seed, trace_variant)
    path = out / filename
    np.savez_compressed(
        path,
        z=np.asarray(run.z, dtype=float),
        endogenous_delta=np.asarray(run.endogenous_delta, dtype=float),
        tau=np.asarray(-1 if run.tau is None else int(run.tau), dtype=int),
        feature_names=np.asarray(run.feature_names, dtype=str),
        scenario=np.asarray(run.scenario, dtype=str),
    )
    _update_trace_manifest(
        out,
        filename,
        source=source,
        env_id=env_id,
        scenario=scenario,
        seed=seed,
        tau=run.tau,
        horizon=int(run.z.shape[0]),
        feature_names=run.feature_names,
        trace_variant=trace_variant,
    )
    return path


def load_trace(
    directory: str | Path,
    *,
    source: str,
    env_id: str,
    scenario: str,
    seed: int,
    trace_variant: str | None = None,
) -> SimulatedRun | None:
    path = Path(directory) / _trace_filename(source, env_id, scenario, seed, trace_variant)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        tau_raw = int(np.asarray(data["tau"]).item())
        feature_names = tuple(str(value) for value in np.asarray(data["feature_names"]).tolist())
        saved_scenario = str(np.asarray(data["scenario"]).item())
        return SimulatedRun(
            z=np.asarray(data["z"], dtype=float),
            endogenous_delta=np.asarray(data["endogenous_delta"], dtype=float),
            tau=None if tau_raw < 0 else tau_raw,
            feature_names=feature_names,
            scenario=saved_scenario,
        )


def _mujoco_trace_variant(config: ExperimentConfig, scenario: str, *, legacy: bool = False) -> str:
    trace_variant = config.mujoco_feature_mode
    if config.mujoco_probe_update_mode != "online":
        trace_variant = f"{trace_variant}_{config.mujoco_probe_update_mode}"
    if config.mujoco_frozen_buffer_size != 96:
        trace_variant = f"{trace_variant}_fb{config.mujoco_frozen_buffer_size}"
    if (not legacy) or config.mujoco_policy_std != 0.35:
        trace_variant = f"{trace_variant}_std{config.mujoco_policy_std:g}"
    if (not legacy) or config.mujoco_policy_mode != "linear_actor":
        trace_variant = f"{trace_variant}_policy{config.mujoco_policy_mode}"
    if config.mujoco_knn_neighbors != 8:
        trace_variant = f"{trace_variant}_knn{config.mujoco_knn_neighbors}"
    if config.mujoco_freeze_task_goal:
        trace_variant = f"{trace_variant}_fixedgoal"
    if scenario == "gravity":
        trace_variant = f"{trace_variant}_g{config.mujoco_gravity_scale:g}"
    elif scenario == "friction":
        trace_variant = f"{trace_variant}_f{config.mujoco_friction_scale:g}"
    elif scenario == "target":
        trace_variant = f"{trace_variant}_t{config.mujoco_target_shift:g}"
    elif scenario == "sensor_bias":
        trace_variant = f"{trace_variant}_sb{config.mujoco_sensor_bias_scale:g}"
    elif scenario == "actuator_loss":
        trace_variant = f"{trace_variant}_al{config.mujoco_actuator_loss_scale:g}"
    return trace_variant


def _mujoco_trace_load_variants(config: ExperimentConfig, scenario: str) -> tuple[str, ...]:
    current = _mujoco_trace_variant(config, scenario)
    legacy = _mujoco_trace_variant(config, scenario, legacy=True)
    return (current,) if current == legacy else (current, legacy)


def run_detector(detector: object, run: SimulatedRun) -> int | None:
    for idx, row in enumerate(run.z):
        delta = run.endogenous_delta[idx]
        result = detector.update(row, endogenous_delta=delta)
        if result.alarm is not None:
            return int(result.alarm.time)
    return None


def generate_experiment_stream(
    config: ExperimentConfig,
    scenario: str,
    seed: int,
    env_id: str | None = None,
) -> SimulatedRun:
    trace_env_id = (env_id or config.mujoco_env_id) if config.source == "mujoco" else "synthetic"
    if config.source == "mujoco":
        trace_variant = _mujoco_trace_variant(config, scenario)
    else:
        trace_variant = None
    trace_load_dir = config.load_traces or config.save_traces
    if trace_load_dir is not None:
        trace_variants = (
            _mujoco_trace_load_variants(config, scenario)
            if config.source == "mujoco"
            else (trace_variant,)
        )
        for candidate_variant in trace_variants:
            loaded = load_trace(
                trace_load_dir,
                source=config.source,
                env_id=trace_env_id,
                scenario=scenario,
                seed=seed,
                trace_variant=candidate_variant,
            )
            if loaded is not None:
                return loaded

    if config.source == "synthetic":
        run = generate_rl_internal_stream(
            horizon=config.horizon,
            tau=config.tau,
            d=config.d,
            scenario=scenario,
            seed=seed,
        )
        if config.save_traces is not None:
            save_trace(
                run,
                config.save_traces,
                source=config.source,
                env_id=trace_env_id,
                scenario=scenario,
                seed=seed,
            )
        return run
    if config.source == "mujoco":
        shift_map = {
            "no_shift": "no_shift",
            "gravity": "gravity",
            "friction": "friction",
            "target": "target",
            "sensor_bias": "sensor_bias",
            "actuator_loss": "actuator_loss",
        }
        if scenario not in shift_map:
            raise ValueError(
                f"MuJoCo source supports scenarios {sorted(shift_map)}, got {scenario!r}"
            )
        run = generate_mujoco_internal_stream(
            MujocoTraceConfig(
                env_id=env_id or config.mujoco_env_id,
                horizon=config.horizon,
                tau=config.tau,
                shift=shift_map[scenario],
                seed=seed,
                gravity_scale=config.mujoco_gravity_scale,
                friction_scale=config.mujoco_friction_scale,
                target_shift=config.mujoco_target_shift,
                sensor_bias_scale=config.mujoco_sensor_bias_scale,
                actuator_loss_scale=config.mujoco_actuator_loss_scale,
                freeze_task_goal=config.mujoco_freeze_task_goal,
                policy_lr=config.mujoco_policy_lr,
                value_lr=config.mujoco_value_lr,
                policy_std=config.mujoco_policy_std,
                policy_mode=config.mujoco_policy_mode,  # type: ignore[arg-type]
                feature_mode=config.mujoco_feature_mode,  # type: ignore[arg-type]
                probe_update_mode=config.mujoco_probe_update_mode,  # type: ignore[arg-type]
                frozen_buffer_size=config.mujoco_frozen_buffer_size,
                knn_neighbors=config.mujoco_knn_neighbors,
            )
        )
        if config.save_traces is not None:
            save_trace(
                run,
                config.save_traces,
                source=config.source,
                env_id=trace_env_id,
                scenario=scenario,
                seed=seed,
                trace_variant=trace_variant,
            )
        return run
    raise ValueError(f"unknown experiment source {config.source!r}")


def detectability_rows_for_stream(
    run: SimulatedRun,
    *,
    scenario: str,
    run_idx: int,
    seed: int,
    env_id: str,
    window: int,
    separation_floor: float = 0.10,
) -> list[dict[str, object]]:
    z = np.asarray(run.z, dtype=float)
    delta = np.asarray(run.endogenous_delta, dtype=float)
    residual = z - np.cumsum(delta, axis=0) if delta.shape == z.shape else z
    horizon, d = residual.shape
    tau = run.tau
    if tau is None:
        split = horizon // 2
        n = min(window, split, horizon - split)
        start_pre = max(0, split - n)
        pre = residual[start_pre:split]
        post = residual[split : split + n]
    else:
        n = min(window, int(tau), horizon - int(tau))
        start_pre = max(0, int(tau) - n)
        pre = residual[start_pre : int(tau)]
        post = residual[int(tau) : int(tau) + n]

    rows: list[dict[str, object]] = []
    feature_names = run.feature_names
    for channel in range(d):
        if pre.size == 0 or post.size == 0:
            scale_sep = np.nan
            dynamics_sep = np.nan
            metric_sep = np.nan
            detectable = False
        else:
            pre_channel = pre[:, channel]
            post_channel = post[:, channel]
            scale_sep = dispersion_ks_distance(pre_channel, post_channel)
            pre_dyn = np.diff(pre_channel)
            post_dyn = np.diff(post_channel)
            dynamics_sep = abs(flip_rate(pre_dyn) - flip_rate(post_dyn))
            metric_sep = max(scale_sep, dynamics_sep)
            detectable = bool(metric_sep >= separation_floor)
        rows.append(
            {
                "env_id": env_id,
                "scenario": scenario,
                "run": int(run_idx),
                "seed": int(seed),
                "tau": tau,
                "channel": channel,
                "feature": feature_names[channel] if channel < len(feature_names) else f"feature_{channel}",
                "scale_separation": scale_sep,
                "dynamics_separation": dynamics_sep,
                "max_metric_separation": metric_sep,
                "separation_floor": separation_floor,
                "detectable": detectable,
                "audit_status": "diagnostic_not_null_calibrated",
                "interpretation": "residual_metric_separation_diagnostic_only",
            }
        )
    return rows


def evaluate_suite(
    config: ExperimentConfig,
    scenarios: list[str] | None = None,
    detector_factories: dict[str, DetectorFactory] | None = None,
    alpha_by_detector: dict[str, float] | None = None,
    include_references: bool = True,
    audit_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    if scenarios is None:
        if config.source == "mujoco":
            scenarios = ["no_shift", "gravity", "friction"]
        else:
            scenarios = ["no_shift", "scale", "dynamics", "mixed", "mean"]
    factories = detector_factories or default_detector_factories(
        config,
        alpha_by_detector,
        include_references=include_references,
    )

    rows: list[dict[str, object]] = []
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
    n_runs = int(config.eval_runs if config.eval_runs is not None else config.n_runs)
    for env_index, env_id in enumerate(_evaluation_env_ids(config)):
        for scenario in scenarios:
            for run_idx in range(n_runs):
                seed = (
                    config.seed
                    + 1009 * run_idx
                    + 9173 * scenario_offsets.get(scenario, 99)
                    + 104729 * env_index
                )
                stream = generate_experiment_stream(config, scenario, seed, env_id=env_id)
                if audit_rows is not None:
                    audit_rows.extend(
                        detectability_rows_for_stream(
                            stream,
                            scenario=scenario,
                            run_idx=run_idx,
                            seed=seed,
                            env_id=env_id,
                            window=config.scale_window,
                        )
                )
                for detector_name, factory in factories.items():
                    detector = factory(stream.z.shape[1])
                    alarm_time = run_detector(detector, stream)
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
                            "run": run_idx,
                            "seed": seed,
                            "detector": detector_name,
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


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["detector", "env_id", "scenario"] if "env_id" in results else ["detector", "scenario"]
    for key, group in results.groupby(group_columns, sort=True):
        if len(group_columns) == 3:
            detector, env_id, scenario = key
        else:
            detector, scenario = key
            env_id = "all"
        has_change = bool(group["has_change"].iloc[0])
        row: dict[str, object] = {
            "detector": detector,
            "env_id": env_id,
            "scenario": scenario,
            "runs": int(group.shape[0]),
            "false_alarm_rate": float(group["false_alarm"].mean()),
        }
        if has_change:
            valid = group[~group["false_alarm"]]
            detected = valid[valid["detected"]]
            row["detection_rate"] = float(group["detected"].mean())
            row["mean_delay_censored"] = float(valid["delay"].mean()) if not valid.empty else np.nan
            row["mean_delay_detected"] = (
                float(detected["delay"].mean()) if not detected.empty else np.nan
            )
            row["median_delay_detected"] = (
                float(detected["delay"].median()) if not detected.empty else np.nan
            )
        else:
            row["detection_rate"] = np.nan
            alarm_times = pd.to_numeric(group["alarm_time"], errors="coerce")
            run_lengths = pd.to_numeric(group["run_length"], errors="coerce")
            time_to_alarm = alarm_times.fillna(run_lengths)
            row["arl0_truncated"] = float(time_to_alarm.mean())
            row["mean_delay_censored"] = np.nan
            row["mean_delay_detected"] = np.nan
            row["median_delay_detected"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_alphas_to_far(
    config: ExperimentConfig,
    target_far: float,
    n_calibration_runs: int | None = None,
    detector_names: list[str] | None = None,
    alpha_bounds: tuple[float, float] = (1e-6, 0.5),
    iterations: int = 10,
    include_references: bool = True,
) -> dict[str, float]:
    """Empirically tune detector alpha values on no-shift streams.

    This provides the matched-FAR comparison required by the paper protocol. It
    uses a monotone bisection heuristic: larger alpha should weakly increase the
    probability of a no-shift alarm.
    """

    names = detector_names or list(
        default_detector_factories(config, include_references=include_references).keys()
    )
    calibrated: dict[str, float] = {}
    calibration_runs = int(n_calibration_runs or config.calibration_runs)
    cal_config = ExperimentConfig(
        **{**config.__dict__, "n_runs": calibration_runs, "eval_runs": calibration_runs}
    )

    for name in names:
        lo, hi = alpha_bounds
        best = lo
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            factories = default_detector_factories(
                cal_config,
                {name: mid},
                include_references=include_references,
            )
            results = evaluate_suite(
                cal_config,
                scenarios=["no_shift"],
                detector_factories={name: factories[name]},
                include_references=include_references,
            )
            far = float(results["false_alarm"].mean())
            if far <= target_far:
                best = mid
                lo = mid
            else:
                hi = mid
        calibrated[name] = best
    return calibrated


def write_summary_markdown(summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    lines = [
        "# Experiment Summary",
        "",
        summary.sort_values(["scenario", "detector"]).to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment_suite(
    config: ExperimentConfig,
    outdir: str | Path,
    scenarios: list[str] | None = None,
    match_far: bool = False,
    target_far: float | None = None,
    include_references: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    alpha_by_detector = None
    if match_far:
        alpha_by_detector = calibrate_alphas_to_far(
            config,
            target_far=config.alpha if target_far is None else target_far,
            n_calibration_runs=config.calibration_runs,
            include_references=include_references,
        )
        pd.DataFrame(
            [
                {
                    "detector": detector,
                    "calibrated_alpha": alpha,
                    "target_far": config.alpha if target_far is None else target_far,
                    "calibration_runs": config.calibration_runs,
                }
                for detector, alpha in sorted(alpha_by_detector.items())
            ]
        ).to_csv(out / "calibrated_alphas.csv", index=False)
    audit_rows: list[dict[str, object]] = []
    results = evaluate_suite(
        config,
        scenarios=scenarios,
        alpha_by_detector=alpha_by_detector,
        include_references=include_references,
        audit_rows=audit_rows,
    )
    summary = summarize_results(results)
    audit = pd.DataFrame(audit_rows)
    ablations = summary[
        summary["detector"].isin(
            ["CADET", "CADET_no_PUR", "CADET_scale", "CADET_dynamics", "CADET_location"]
        )
    ]
    results.to_csv(out / "raw_results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    audit.to_csv(out / "detectability_audit.csv", index=False)
    ablations.to_csv(out / "ablation_summary.csv", index=False)
    write_summary_markdown(summary, out / "summary.md")
    manifest: dict[str, object] = {
        "source": config.source,
        "env_ids": list(_evaluation_env_ids(config)),
        "save_traces": config.save_traces,
        "load_traces": config.load_traces,
    }
    trace_dir = config.save_traces or config.load_traces
    if trace_dir is not None:
        trace_manifest = Path(trace_dir) / "trace_manifest.json"
        if trace_manifest.exists():
            manifest["traces"] = json.loads(trace_manifest.read_text(encoding="utf-8")).get(
                "traces",
                [],
            )
    (out / "trace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return results, summary
