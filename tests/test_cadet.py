import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import cadet.calibration as calibration_module
from cadet.baselines import (
    BOCPDGaussianDetector,
    BaselineConfig,
    EnergyDistanceDetector,
    KernelMMDDetector,
    PageHinkleyDetector,
)
from cadet.calibration import (
    calibrate_from_run_maxima,
    empirical_run_threshold,
    evaluate_heldout_far,
    evaluate_empirical_threshold_suite,
    run_empirical_calibration,
    run_detector_with_empirical_threshold,
    statistic_rows_for_run,
)
from cadet.detectors import (
    AnchoredLocationArm,
    CADETConfig,
    CADETDetector,
    LocationArm,
    ScaleArm,
)
from cadet.evaluation import (
    ExperimentConfig,
    detectability_rows_for_stream,
    generate_experiment_stream,
    load_trace,
    run_experiment_suite,
    save_trace,
)
from cadet.mujoco import (
    _LinearActorValue,
    MujocoTraceConfig,
    _FrozenForceBalanceResidualProbe,
    _FrozenGravityCompensationProbe,
    _FrozenGravityResidualProbe,
    _FrozenGaitLoadResponseProbe,
    _FrozenGaitPhaseResponseProbe,
    _FrozenKernelSupportResponseProbe,
    _FrozenPhysicsResidualProbe,
    _FrozenWalkerExcitationResponseProbe,
    _FrozenWalkerGravityResponseProbe,
    _FrozenWalkerInnovationGainProbe,
    _Transition,
    _actuator_efficiency_probe_feature_vector,
    _apply_actuator_loss,
    _contact_slip_probe_feature_vector,
    _locomotion_dynamics_probe_feature_vector,
    _locomotion_observation_probe_feature_vector,
    _locomotion_stability_probe_feature_vector,
    _linear_probe_burst_action,
    _mechanical_energy_probe_feature_vector,
    _motion_probe_feature_vector,
    _force_balance_probe_target_vector,
    _gravity_balance_probe_feature_vector,
    _gravity_compensation_probe_target_vector,
    _gravity_signature_probe_feature_vector,
    _walker_excitation_response_probe_target_vector,
    _walker_gravity_response_probe_target_vector,
    _walker_innovation_gain_probe_target_vector,
    _physics_probe_target_vector,
    _sensor_bias_vector,
    _scripted_policy_action,
    _support_response_probe_target_vector,
    generate_mujoco_internal_stream,
    mujoco_feature_names,
)
from cadet.residualization import (
    FrozenBufferPURResidualizer,
    IdentityResidualizer,
    LinearPURResidualizer,
)
from cadet.simulation import SimulatedRun, generate_rl_internal_stream
from cadet.utils import P2Quantile, dispersion_ks_distance, flip_rate, robust_whittle_ar1_fit


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_dispersion_ks_distance_responds_to_scale_change():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=200)
    detection = 3.5 * rng.normal(size=200)
    assert dispersion_ks_distance(reference, detection) > 0.25


def test_scale_arm_fires_on_large_scale_shift():
    rng = np.random.default_rng(11)
    arm = ScaleArm(window=60, lambda_phi=1.0)
    result = None
    for value in rng.normal(size=120):
        result = arm.update(value, alpha=0.2)
    for value in 4.0 * rng.normal(size=60):
        result = arm.update(value, alpha=0.2)
    assert result is not None
    assert result.ready
    assert result.fired


def test_location_arm_fires_on_large_mean_shift():
    arm = LocationArm(window=20, lambda_phi=1.0)
    result = None
    for value in [0.0] * 20 + [2.0] * 20:
        result = arm.update(value, alpha=0.5)
    assert result is not None
    assert result.ready
    assert result.fired


def test_location_arm_trims_sparse_outliers():
    arm = LocationArm(window=20, lambda_phi=1.0)
    result = None
    for value in [0.0] * 20 + [0.0] * 18 + [100.0] * 2:
        result = arm.update(value, alpha=0.5)
    assert result is not None
    assert result.ready
    assert not result.fired
    assert result.metadata["trim_fraction"] == 0.1


def test_anchored_location_arm_keeps_initial_reference():
    arm = AnchoredLocationArm(window=10, lambda_phi=1.0)
    first = None
    second = None
    for value in [0.0] * 10 + [1.0] * 10 + [1.0] * 10:
        result = arm.update(value, alpha=0.5)
        if result.ready and first is None:
            first = result
        elif result.ready:
            second = result
    assert first is not None
    assert second is not None
    assert first.fired
    assert second.fired
    assert second.metadata["anchored"] is True
    assert second.metadata["reference_location"] == pytest.approx(0.0)


def test_scale_arm_uses_frozen_reference_center():
    arm = ScaleArm(window=10, lambda_phi=1.0, center_mode="reference")
    first_result = None
    for value in np.linspace(-1.0, 1.0, 10):
        first_result = arm.update(float(value), alpha=0.2)
    assert first_result is not None
    reference_center = first_result.metadata["reference_center"]

    second_result = None
    for value in np.linspace(4.0, 6.0, 10):
        second_result = arm.update(float(value), alpha=0.2)
    assert second_result is not None
    assert second_result.metadata["center_mode"] == "reference"
    assert second_result.metadata["reference_center"] == reference_center


def test_scale_arm_iid_null_is_conservative_with_reference_center():
    rng = np.random.default_rng(19)
    arm = ScaleArm(window=80, lambda_phi=1.0, center_mode="reference")
    fired = False
    for value in rng.normal(size=400):
        result = arm.update(float(value), alpha=1e-3)
        fired = fired or (result.ready and result.fired)
    assert not fired


def test_flip_rate_tracks_alternating_dynamics():
    alternating = np.asarray([1.0, -1.0] * 50)
    smooth = np.ones(100)
    assert flip_rate(alternating) > 0.95
    assert flip_rate(smooth) == 0.0


def test_p2_quantile_tracks_median():
    rng = np.random.default_rng(17)
    estimator = P2Quantile(0.5)
    values = rng.standard_t(df=5, size=500)
    for value in values:
        estimator.update(value)
    assert abs(estimator.value - np.median(values)) < 0.12


def test_frozen_buffer_pur_removes_linear_update():
    buffer = np.asarray([1.0, 2.0, 3.0])

    def feature_fn(states, theta):
        return states[:, None] * theta[None, :]

    residualizer = FrozenBufferPURResidualizer(
        feature_fn=feature_fn,
        params=np.asarray([1.0, -1.0]),
        buffer=buffer,
    )
    delta = np.asarray([0.1, -0.2])
    z = feature_fn(buffer, np.asarray([1.1, -1.2])).mean(axis=0)
    residual = residualizer.update(z, endogenous_delta=delta)
    assert np.linalg.norm(residual) < 1e-10


def test_robust_whittle_uses_fisher_refinement():
    rng = np.random.default_rng(23)
    phi_true = 0.6
    values = []
    current = 0.0
    for _ in range(256):
        current = phi_true * current + np.sqrt(1.0 - phi_true**2) * rng.normal()
        values.append(current)

    phi, sigma2, objective, info = robust_whittle_ar1_fit(
        np.asarray(values),
        return_info=True,
    )
    assert info["solver_status"].startswith("fisher")
    assert objective <= info["grid_objective"] + 1e-8
    assert abs(phi - phi_true) < 0.2
    assert sigma2 > 0.0


def test_mujoco_backend_reports_missing_optional_dependency():
    if importlib.util.find_spec("gymnasium") is not None:
        pytest.skip("Gymnasium installed; missing-dep path not applicable")
    with pytest.raises(ModuleNotFoundError, match="gymnasium\\[mujoco\\]"):
        generate_mujoco_internal_stream(MujocoTraceConfig(horizon=2, tau=1))


def test_mujoco_feature_modes_are_explicit():
    actor_names = mujoco_feature_names("actor_value")
    rich_names = mujoco_feature_names("actor_value_kinematic")
    telemetry_names = mujoco_feature_names("actor_value_telemetry")
    domain_names = mujoco_feature_names("actor_value_domain")
    reset_probe_names = mujoco_feature_names("actor_value_reset_probe")
    reset_only_names = mujoco_feature_names("reset_probe")
    model_probe_names = mujoco_feature_names("model_probe")
    model_state_probe_names = mujoco_feature_names("model_state_probe")
    model_vector_probe_names = mujoco_feature_names("model_vector_probe")
    motion_probe_names = mujoco_feature_names("motion_probe")
    task_reward_probe_names = mujoco_feature_names("task_reward_probe")
    task_observation_probe_names = mujoco_feature_names("task_observation_probe")
    knn_state_probe_names = mujoco_feature_names("knn_state_probe")
    phase_state_probe_names = mujoco_feature_names("phase_state_probe")
    gravity_signature_probe_names = mujoco_feature_names("gravity_signature_probe")
    gravity_residual_probe_names = mujoco_feature_names("gravity_residual_probe")
    gravity_compensation_probe_names = mujoco_feature_names("gravity_compensation_probe")
    gravity_balance_probe_names = mujoco_feature_names("gravity_balance_probe")
    walker_gravity_response_probe_names = mujoco_feature_names(
        "walker_gravity_response_probe"
    )
    walker_excitation_response_probe_names = mujoco_feature_names(
        "walker_excitation_response_probe"
    )
    walker_innovation_gain_probe_names = mujoco_feature_names(
        "walker_innovation_gain_probe"
    )
    gait_phase_response_probe_names = mujoco_feature_names("gait_phase_response_probe")
    gait_load_response_probe_names = mujoco_feature_names("gait_load_response_probe")
    mechanical_energy_probe_names = mujoco_feature_names("mechanical_energy_probe")
    control_response_probe_names = mujoco_feature_names("control_response_probe")
    actuator_efficiency_probe_names = mujoco_feature_names("actuator_efficiency_probe")
    sensor_consistency_probe_names = mujoco_feature_names("sensor_consistency_probe")
    physics_residual_probe_names = mujoco_feature_names("physics_residual_probe")
    force_balance_probe_names = mujoco_feature_names("force_balance_probe")
    contact_slip_probe_names = mujoco_feature_names("contact_slip_probe")
    support_telemetry_probe_names = mujoco_feature_names("support_telemetry_probe")
    support_response_probe_names = mujoco_feature_names("support_response_probe")
    kernel_support_response_probe_names = mujoco_feature_names("kernel_support_response_probe")

    assert rich_names[: len(actor_names)] == actor_names
    assert {"reward", "obs_delta_norm", "qvel_norm"} <= set(rich_names)
    assert len(rich_names) > len(actor_names)
    assert telemetry_names[: len(rich_names)] == rich_names
    assert {"x_velocity", "x_position", "reward_forward", "reward_ctrl"} <= set(telemetry_names)
    assert len(telemetry_names) > len(rich_names)
    assert domain_names[: len(rich_names)] == rich_names
    assert {"x_velocity", "reward_forward", "reward_ctrl"} <= set(domain_names)
    assert "x_position" not in domain_names
    assert reset_probe_names[: len(domain_names)] == domain_names
    assert {"probe_reward_delta", "probe_obs_delta_norm", "probe_qvel_delta_norm"} <= set(
        reset_probe_names
    )
    assert reset_only_names == (
        "probe_reward_delta",
        "probe_obs_delta_norm",
        "probe_qpos_delta_norm",
        "probe_qvel_delta_norm",
    )
    assert model_probe_names == (
        "model_reward_abs_error",
        "model_obs_delta_error_norm",
        "model_qpos_delta_error_norm",
        "model_qvel_delta_error_norm",
    )
    assert model_state_probe_names == model_probe_names[1:]
    assert model_vector_probe_names == (
        "model_obs_resid_mean",
        "model_obs_resid_alt",
        "model_obs_resid_ramp",
        "model_qpos_resid_mean",
        "model_qpos_resid_alt",
        "model_qpos_resid_ramp",
        "model_qvel_resid_mean",
        "model_qvel_resid_alt",
        "model_qvel_resid_ramp",
    )
    assert motion_probe_names == (
        "motion_obs_delta_norm",
        "motion_qpos_delta_rel_norm",
        "motion_qvel_delta_norm",
        "motion_qvel_delta_max_abs",
    )
    assert task_reward_probe_names == (
        "task_reward",
        "task_reward_dist",
        "task_reward_near",
        "task_reward_forward",
    )
    assert task_observation_probe_names == (
        "task_obs_coord_0",
        "task_obs_coord_1",
        "task_obs_coord_norm",
        "task_obs_coord_delta_norm",
    )
    assert knn_state_probe_names == (
        "knn_obs_delta_error_norm",
        "knn_qpos_delta_rel_error_norm",
        "knn_qvel_delta_error_norm",
    )
    assert phase_state_probe_names == (
        "phase_obs_delta_error_norm",
        "phase_qpos_delta_rel_error_norm",
        "phase_qvel_delta_error_norm",
    )
    assert len(gravity_signature_probe_names) == 26
    assert gravity_signature_probe_names[0] == "gravity_qfrc_bias_norm"
    assert "gravity_vertical_balance_proxy" in gravity_signature_probe_names
    assert gravity_signature_probe_names[-1] == "gravity_abs_height_delta_per_action"
    assert len(gravity_residual_probe_names) == len(gravity_signature_probe_names)
    assert gravity_residual_probe_names[0] == "gravity_qfrc_bias_norm_resid"
    assert "gravity_vertical_balance_proxy_resid" in gravity_residual_probe_names
    assert gravity_residual_probe_names[-1] == "gravity_abs_height_delta_per_action_resid"
    assert len(gravity_compensation_probe_names) == 18
    assert gravity_compensation_probe_names[0] == "gravity_comp_qfrc_bias_resid_00"
    assert "gravity_comp_bias_per_height_resid" in gravity_compensation_probe_names
    assert gravity_compensation_probe_names[-1] == "gravity_comp_bias_passive_alignment_resid"
    assert len(gravity_balance_probe_names) == 18
    assert gravity_balance_probe_names[0] == "gravity_balance_log_load"
    assert "gravity_balance_total_vertical_per_load" in gravity_balance_probe_names
    assert gravity_balance_probe_names[-1] == "gravity_balance_load_concentration"
    assert len(walker_gravity_response_probe_names) == 24
    assert walker_gravity_response_probe_names[0] == "walker_gravity_bias_vertical_resid"
    assert "walker_gravity_total_vertical_balance_resid" in walker_gravity_response_probe_names
    assert walker_gravity_response_probe_names[-1] == "walker_gravity_action_vertical_response_resid"
    assert len(walker_excitation_response_probe_names) == 25
    assert walker_excitation_response_probe_names[0] == "walker_excitation_root_forward_delta_resid"
    assert "walker_excitation_vertical_per_innovation_resid" in walker_excitation_response_probe_names
    assert walker_excitation_response_probe_names[-1] == "walker_excitation_abs_action_delta_joint_alignment_resid"
    assert len(walker_innovation_gain_probe_names) == 25
    assert walker_innovation_gain_probe_names[0] == "walker_gain_forward_sum_resid"
    assert "walker_gain_pitch_phase_resid_00" in walker_innovation_gain_probe_names
    assert walker_innovation_gain_probe_names[-1] == "walker_gain_innovation_joint_alignment_resid"
    assert len(gait_phase_response_probe_names) == 24
    assert gait_phase_response_probe_names[0] == "gait_leg_velocity_resid_00"
    assert "gait_forward_per_action_resid" in gait_phase_response_probe_names
    assert gait_phase_response_probe_names[-1] == "gait_pitch_rate_per_action_resid"
    assert len(gait_load_response_probe_names) == 36
    assert gait_load_response_probe_names[0] == "gait_load_total_resid"
    assert "gait_bias_qacc_alignment_resid" in gait_load_response_probe_names
    assert gait_load_response_probe_names[-1] == "gait_loaded_speed_share_resid_07"
    assert len(mechanical_energy_probe_names) == 18
    assert mechanical_energy_probe_names[0] == "mechanical_kinetic_proxy"
    assert mechanical_energy_probe_names[-1] == "mechanical_reward_per_action"
    assert len(control_response_probe_names) == 34
    assert control_response_probe_names[0] == "control_qvel_delta_resid_00"
    assert "control_forward_per_action_resid" in control_response_probe_names
    assert len(actuator_efficiency_probe_names) == 13
    assert actuator_efficiency_probe_names[0] == "actuator_action_norm"
    assert actuator_efficiency_probe_names[-1] == "actuator_angle_delta"
    assert len(sensor_consistency_probe_names) == 17
    assert sensor_consistency_probe_names[0] == "sensor_obs_resid_00"
    assert sensor_consistency_probe_names[-1] == "sensor_obs_resid_16"
    assert len(physics_residual_probe_names) == 35
    assert physics_residual_probe_names[0] == "physics_cfrc_norm_resid_00"
    assert physics_residual_probe_names[-1] == "physics_qacc_resid_08"
    assert len(force_balance_probe_names) == 93
    assert force_balance_probe_names[0] == "force_cfrc_ext_resid_00"
    assert "force_qfrc_bias_resid_00" in force_balance_probe_names
    assert force_balance_probe_names[-1] == "force_qacc_resid_08"
    assert len(contact_slip_probe_names) == 54
    assert contact_slip_probe_names[0] == "contact_count"
    assert "contact_loaded_xy_speed_00" in contact_slip_probe_names
    assert contact_slip_probe_names[-1] == "contact_low_height_xy_speed_07"
    assert len(support_response_probe_names) == 36
    assert support_response_probe_names[0] == "support_load_total_resid"
    assert "support_bias_qacc_alignment_resid" in support_response_probe_names
    assert support_response_probe_names[-1] == "support_loaded_speed_share_resid_07"
    assert len(support_telemetry_probe_names) == len(support_response_probe_names)
    assert support_telemetry_probe_names[0] == "support_load_total"
    assert "support_bias_qacc_alignment" in support_telemetry_probe_names
    assert support_telemetry_probe_names[-1] == "support_loaded_speed_share_07"
    assert len(kernel_support_response_probe_names) == len(support_response_probe_names)
    assert kernel_support_response_probe_names[0] == "kernel_support_load_total_resid"
    assert "kernel_support_bias_qacc_alignment_resid" in kernel_support_response_probe_names
    assert kernel_support_response_probe_names[-1] == "kernel_support_loaded_speed_share_resid_07"
    with pytest.raises(ValueError, match="unknown MuJoCo feature mode"):
        MujocoTraceConfig(feature_mode="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown MuJoCo shift"):
        MujocoTraceConfig(shift="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown MuJoCo probe update mode"):
        MujocoTraceConfig(probe_update_mode="bad")  # type: ignore[arg-type]
    assert (
        MujocoTraceConfig(policy_mode="linear_sinusoidal_dither").policy_mode
        == "linear_sinusoidal_dither"
    )
    assert MujocoTraceConfig(policy_mode="linear_probe_burst").policy_mode == "linear_probe_burst"
    assert MujocoTraceConfig(policy_mode="chirp").policy_mode == "chirp"
    assert MujocoTraceConfig(policy_mode="pulse").policy_mode == "pulse"
    with pytest.raises(ValueError, match="unknown MuJoCo policy mode"):
        MujocoTraceConfig(policy_mode="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sensor_bias_scale must be positive"):
        MujocoTraceConfig(sensor_bias_scale=0.0)
    with pytest.raises(ValueError, match="actuator_loss_scale must be in"):
        MujocoTraceConfig(actuator_loss_scale=0.0)
    with pytest.raises(ValueError, match="knn_neighbors must be positive"):
        MujocoTraceConfig(knn_neighbors=0)


def test_sensor_bias_vector_is_deterministic_and_bounded():
    first = _sensor_bias_vector(6, 0.35, 123)
    second = _sensor_bias_vector(6, 0.35, 123)
    changed_seed = _sensor_bias_vector(6, 0.35, 124)

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, changed_seed)
    assert first.shape == (6,)
    assert np.max(np.abs(first)) == pytest.approx(0.35)


def test_actuator_loss_scales_executed_action_only():
    command = np.asarray([0.8, -0.4, 0.0])
    executed = _apply_actuator_loss(command, 0.55)
    np.testing.assert_allclose(executed, np.asarray([0.44, -0.22, 0.0]))
    np.testing.assert_allclose(command, np.asarray([0.8, -0.4, 0.0]))


def test_mujoco_scripted_policy_modes_are_finite_bounded_and_distinct():
    action_low = -np.ones(6, dtype=float)
    action_high = np.ones(6, dtype=float)
    actions = {
        mode: _scripted_policy_action(
            mode,  # type: ignore[arg-type]
            t=37,
            action_low=action_low,
            action_high=action_high,
            env_id="HalfCheetah-v5",
        )
        for mode in ("zero", "sinusoidal", "chirp", "pulse")
    }

    assert np.allclose(actions["zero"], 0.0)
    for action in actions.values():
        assert action.shape == action_low.shape
        assert np.all(np.isfinite(action))
        assert np.all(action >= action_low)
        assert np.all(action <= action_high)
    assert not np.allclose(actions["sinusoidal"], actions["chirp"])
    assert not np.allclose(actions["sinusoidal"], actions["pulse"])
    assert not np.allclose(actions["chirp"], actions["pulse"])
    with pytest.raises(ValueError, match="unsupported policy mode"):
        _scripted_policy_action(  # type: ignore[arg-type]
            "linear_actor",
            t=37,
            action_low=action_low,
            action_high=action_high,
            env_id="HalfCheetah-v5",
        )


def test_linear_probe_burst_is_sparse_bounded_and_actor_based():
    action_low = -np.ones(6, dtype=float)
    action_high = np.ones(6, dtype=float)
    agent = _LinearActorValue(
        obs_dim=17,
        action_dim=6,
        hidden_dim=8,
        action_low=action_low,
        action_high=action_high,
        policy_std=0.0,
        seed=123,
    )
    obs = np.linspace(-0.2, 0.2, 17)
    base = agent.act(obs)
    quiet = _linear_probe_burst_action(
        agent,
        obs,
        t=40,
        action_low=action_low,
        action_high=action_high,
        env_id="Walker2d-v5",
    )
    burst = _linear_probe_burst_action(
        agent,
        obs,
        t=8,
        action_low=action_low,
        action_high=action_high,
        env_id="Walker2d-v5",
    )

    assert quiet.shape == action_low.shape
    assert burst.shape == action_low.shape
    assert np.all(np.isfinite(quiet))
    assert np.all(np.isfinite(burst))
    assert np.all(quiet >= action_low)
    assert np.all(quiet <= action_high)
    assert np.all(burst >= action_low)
    assert np.all(burst <= action_high)
    assert np.allclose(quiet, base)
    assert not np.allclose(burst, base)


def test_anytime_clock_does_not_reset_after_alarm():
    detector = CADETDetector(
        CADETConfig(
            alpha=0.99,
            scale_window=5,
            dynamics_window=5,
            warmup=0,
            anytime=False,
            arms=("scale",),
            reset_after_alarm=True,
        ),
        residualizer=IdentityResidualizer(),
    )
    first_alarm_time = None
    for value in list(np.zeros(10)) + list(np.full(10, 10.0)):
        result = detector.update(np.asarray([value]), endogenous_delta=np.asarray([0.0]))
        if result.alarm is not None:
            first_alarm_time = result.time
            break
    assert first_alarm_time is not None
    assert detector.t == first_alarm_time
    assert detector.segment_t == 0

    result = detector.update(np.asarray([0.0]), endogenous_delta=np.asarray([0.0]))
    assert result.time == first_alarm_time + 1
    assert detector.t == first_alarm_time + 1
    assert detector.segment_t == 1


def test_new_baselines_fire_on_simple_mean_shift():
    detectors = [
        PageHinkleyDetector,
        KernelMMDDetector,
        EnergyDistanceDetector,
        BOCPDGaussianDetector,
    ]
    stream = list(np.zeros(25)) + list(np.full(40, 3.0))
    for detector_cls in detectors:
        detector = detector_cls(BaselineConfig(alpha=0.5, window=10, warmup=20, anytime=False))
        alarm_time = None
        for value in stream:
            result = detector.update(np.asarray([value]), endogenous_delta=np.asarray([0.0]))
            if result.alarm is not None:
                alarm_time = result.time
                break
        assert alarm_time is not None, detector_cls.__name__
        assert alarm_time >= 20


def test_baseline_step_result_reports_candidate_statistics():
    detector = PageHinkleyDetector(
        BaselineConfig(alpha=1e-300, window=10, warmup=5, anytime=False)
    )
    last_result = None
    for value in np.linspace(0.0, 1.0, 20):
        last_result = detector.update(np.asarray([value]), endogenous_delta=np.asarray([0.0]))
    assert last_result is not None
    assert last_result.statistics
    assert last_result.alarms == []
    assert last_result.statistics[0].arm == "page_hinkley"


def test_empirical_threshold_uses_conservative_higher_quantile():
    threshold = empirical_run_threshold([1.0, 2.0, 3.0, 4.0], target_far=0.25)
    assert threshold == 4.0
    threshold = empirical_run_threshold([1.0, 2.0, 3.0, 4.0], target_far=0.50)
    assert threshold == 3.0


def test_empirical_calibration_produces_required_tables():
    calibration_config = ExperimentConfig(
        alpha=0.10,
        horizon=80,
        tau=40,
        n_runs=2,
        eval_runs=2,
        calibration_runs=2,
        d=2,
        scale_window=10,
        dynamics_window=12,
        warmup=20,
        anytime=False,
        seed=123,
    )
    heldout_config = ExperimentConfig(
        **{**calibration_config.__dict__, "seed": 456}
    )
    result = run_empirical_calibration(
        calibration_config,
        heldout_config,
        ["ADWIN", "MMD"],
        target_far=0.10,
    )
    assert not result.raw_statistics.empty
    assert {"detector", "statistic", "split", "time"} <= set(result.raw_statistics.columns)
    assert {"detector", "empirical_threshold"} <= set(result.thresholds.columns)
    assert {"detector", "heldout_far", "far_ci_lower", "far_ci_upper"} <= set(
        result.heldout_far.columns
    )
    assert set(result.thresholds["detector"]) == {"ADWIN", "MMD"}
    assert set(result.detector_status["status"]) <= {"admissible", "diagnostic_only", "excluded"}


def test_empirical_threshold_suite_evaluates_shifted_runs():
    config = ExperimentConfig(
        alpha=0.10,
        horizon=90,
        tau=45,
        n_runs=1,
        eval_runs=1,
        calibration_runs=1,
        d=1,
        scale_window=10,
        dynamics_window=12,
        warmup=20,
        anytime=False,
        seed=999,
    )
    thresholds = pd.DataFrame(
        [{"detector": "ADWIN", "empirical_threshold": 10.0}]
    )
    audit_calls = []
    results = evaluate_empirical_threshold_suite(
        config,
        ["no_shift", "mean"],
        thresholds,
        detector_names=["ADWIN"],
        audit_callback=lambda stream, scenario, run_idx, seed, env_id: audit_calls.append(
            (stream.scenario, scenario, run_idx, seed, env_id)
        ),
    )
    assert set(results["scenario"]) == {"no_shift", "mean"}
    assert {"threshold_mode", "max_statistic", "empirical_threshold"} <= set(results.columns)
    assert len(audit_calls) == 2
    assert {call[1] for call in audit_calls} == {"no_shift", "mean"}


def test_empirical_statistic_thresholds_ignore_warmup_suppressed_candidates():
    config = ExperimentConfig(
        alpha=0.10,
        horizon=55,
        tau=40,
        n_runs=1,
        eval_runs=1,
        calibration_runs=1,
        d=1,
        scale_window=5,
        dynamics_window=6,
        warmup=30,
        anytime=False,
        seed=999,
    )
    values = np.zeros((config.horizon, 1), dtype=float)
    values[8:13, 0] = 1000.0
    stream = SimulatedRun(
        z=values,
        endogenous_delta=np.zeros_like(values),
        tau=None,
        feature_names=("probe",),
        scenario="synthetic_no_shift",
    )

    rows = statistic_rows_for_run(
        config,
        "CADET_location",
        stream,
        split="calibration",
        run_idx=0,
        seed=123,
    )
    assert rows
    assert min(row["time"] for row in rows) > config.warmup

    alarm_time, max_statistic = run_detector_with_empirical_threshold(
        config,
        "CADET_location",
        stream,
        threshold=1.0,
    )
    assert alarm_time is None
    assert max_statistic <= 1.0


def test_empirical_threshold_suite_respects_mujoco_env_ids(monkeypatch):
    calls = []

    def fake_stream(config, scenario, seed, env_id=None):
        calls.append((scenario, seed, env_id))
        tau = None if scenario == "no_shift" else config.tau
        values = np.linspace(0.0, 1.0, config.horizon, dtype=float)[:, None]
        return SimulatedRun(
            z=values,
            endogenous_delta=np.zeros_like(values),
            tau=tau,
            feature_names=("td_abs",),
            scenario=f"mujoco_{env_id}_{scenario}",
        )

    monkeypatch.setattr(calibration_module, "generate_experiment_stream", fake_stream)
    config = ExperimentConfig(
        alpha=0.10,
        horizon=80,
        tau=40,
        n_runs=1,
        eval_runs=1,
        calibration_runs=1,
        d=1,
        scale_window=10,
        dynamics_window=12,
        warmup=20,
        anytime=False,
        seed=999,
        source="mujoco",
        mujoco_env_ids=("HalfCheetah-v5", "Walker2d-v5"),
    )
    thresholds = pd.DataFrame([{"detector": "ADWIN", "empirical_threshold": 10.0}])
    results = evaluate_empirical_threshold_suite(
        config,
        ["no_shift", "gravity"],
        thresholds,
        detector_names=["ADWIN"],
    )
    assert set(results["env_id"]) == {"HalfCheetah-v5", "Walker2d-v5"}
    assert set(results["scenario"]) == {"no_shift", "gravity"}
    assert {call[2] for call in calls} == {"HalfCheetah-v5", "Walker2d-v5"}


def test_trace_persistence_round_trip(tmp_path):
    run = generate_rl_internal_stream(horizon=40, tau=20, d=2, scenario="scale", seed=101)
    save_trace(run, tmp_path, source="synthetic", env_id="synthetic", scenario="scale", seed=101)
    loaded = load_trace(
        tmp_path,
        source="synthetic",
        env_id="synthetic",
        scenario="scale",
        seed=101,
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded.z, run.z)
    np.testing.assert_allclose(loaded.endogenous_delta, run.endogenous_delta)
    assert loaded.tau == run.tau
    assert loaded.feature_names == run.feature_names
    assert (tmp_path / "trace_manifest.json").exists()


def test_empirical_calibration_can_group_thresholds_by_environment():
    maxima = pd.DataFrame(
        [
            {"detector": "D", "env_id": "A", "max_statistic": 1.0},
            {"detector": "D", "env_id": "A", "max_statistic": 2.0},
            {"detector": "D", "env_id": "B", "max_statistic": 10.0},
            {"detector": "D", "env_id": "B", "max_statistic": 20.0},
        ]
    )

    thresholds = calibrate_from_run_maxima(
        maxima,
        ["D"],
        target_far=0.05,
        group_by_env=True,
    )
    assert set(thresholds["env_id"]) == {"A", "B"}
    assert float(thresholds.loc[thresholds["env_id"] == "A", "empirical_threshold"].iloc[0]) == 2.0
    assert float(thresholds.loc[thresholds["env_id"] == "B", "empirical_threshold"].iloc[0]) == 20.0

    heldout = pd.DataFrame(
        [
            {"detector": "D", "env_id": "A", "max_statistic": 1.5},
            {"detector": "D", "env_id": "B", "max_statistic": 30.0},
        ]
    )
    heldout_far, status = evaluate_heldout_far(heldout, thresholds, target_far=0.05)
    assert int(heldout_far.loc[heldout_far["env_id"] == "A", "heldout_false_alarms"].iloc[0]) == 0
    assert int(heldout_far.loc[heldout_far["env_id"] == "B", "heldout_false_alarms"].iloc[0]) == 1
    assert status.loc[status["env_id"] == "A", "status"].iloc[0] == "admissible"
    assert status.loc[status["env_id"] == "B", "status"].iloc[0] == "diagnostic_only"


def test_trace_persistence_separates_mujoco_feature_variants(tmp_path):
    base = SimulatedRun(
        z=np.zeros((4, 1)),
        endogenous_delta=np.zeros((4, 1)),
        tau=None,
        feature_names=("base",),
        scenario="mujoco_fake_no_shift",
    )
    rich = SimulatedRun(
        z=np.ones((4, 2)),
        endogenous_delta=np.zeros((4, 2)),
        tau=None,
        feature_names=("base", "reward"),
        scenario="mujoco_fake_no_shift",
    )
    save_trace(
        base,
        tmp_path,
        source="mujoco",
        env_id="FakeEnv-v0",
        scenario="no_shift",
        seed=5,
        trace_variant="actor_value",
    )
    save_trace(
        rich,
        tmp_path,
        source="mujoco",
        env_id="FakeEnv-v0",
        scenario="no_shift",
        seed=5,
        trace_variant="actor_value_kinematic",
    )
    save_trace(
        rich,
        tmp_path,
        source="mujoco",
        env_id="FakeEnv-v0",
        scenario="no_shift",
        seed=5,
        trace_variant="actor_value_kinematic_frozen_policy",
    )

    loaded_base = load_trace(
        tmp_path,
        source="mujoco",
        env_id="FakeEnv-v0",
        scenario="no_shift",
        seed=5,
        trace_variant="actor_value",
    )
    loaded_rich = load_trace(
        tmp_path,
        source="mujoco",
        env_id="FakeEnv-v0",
        scenario="no_shift",
        seed=5,
        trace_variant="actor_value_kinematic",
    )
    loaded_frozen = load_trace(
        tmp_path,
        source="mujoco",
        env_id="FakeEnv-v0",
        scenario="no_shift",
        seed=5,
        trace_variant="actor_value_kinematic_frozen_policy",
    )
    assert loaded_base is not None and loaded_base.z.shape[1] == 1
    assert loaded_rich is not None and loaded_rich.z.shape[1] == 2
    assert loaded_frozen is not None and loaded_frozen.z.shape[1] == 2


def test_mujoco_trace_cache_separates_shift_magnitudes(tmp_path, monkeypatch):
    import cadet.evaluation as evaluation_module

    generated = []

    def fake_mujoco_stream(config):
        generated.append(
            (
                config.shift,
                config.gravity_scale,
                config.friction_scale,
                config.target_shift,
                config.sensor_bias_scale,
                config.actuator_loss_scale,
                config.freeze_task_goal,
                config.policy_mode,
            )
        )
        policy_offset = {
            "linear_actor": 0.0,
            "linear_sinusoidal_dither": 5.0,
            "linear_probe_burst": 7.0,
            "sinusoidal": 10.0,
            "chirp": 20.0,
            "pulse": 30.0,
        }.get(config.policy_mode, 0.0)
        values = np.full(
            (4, 1),
            float(
                config.gravity_scale
                + config.friction_scale
                + config.target_shift
                + config.sensor_bias_scale
                + config.actuator_loss_scale
                + int(config.freeze_task_goal)
                + policy_offset
            ),
        )
        return SimulatedRun(
            z=values,
            endogenous_delta=np.zeros_like(values),
            tau=config.tau,
            feature_names=("probe",),
            scenario=f"mujoco_{config.env_id}_{config.shift}",
        )

    monkeypatch.setattr(evaluation_module, "generate_mujoco_internal_stream", fake_mujoco_stream)
    config = ExperimentConfig(
        source="mujoco",
        horizon=4,
        tau=2,
        save_traces=str(tmp_path),
        mujoco_env_ids=("FakeEnv-v0",),
        mujoco_feature_mode="actor_value_kinematic",
        mujoco_probe_update_mode="frozen_policy",
        mujoco_gravity_scale=1.5,
        mujoco_policy_std=0.1,
        mujoco_frozen_buffer_size=16,
        mujoco_knn_neighbors=3,
    )
    first = generate_experiment_stream(config, "gravity", 10, env_id="FakeEnv-v0")
    second_config = ExperimentConfig(**{**config.__dict__, "mujoco_gravity_scale": 2.0})
    second = generate_experiment_stream(second_config, "gravity", 10, env_id="FakeEnv-v0")
    target_first = generate_experiment_stream(config, "target", 10, env_id="FakeEnv-v0")
    target_config = ExperimentConfig(**{**config.__dict__, "mujoco_target_shift": 0.25})
    target_second = generate_experiment_stream(
        target_config,
        "target",
        10,
        env_id="FakeEnv-v0",
    )
    sensor_first = generate_experiment_stream(config, "sensor_bias", 10, env_id="FakeEnv-v0")
    sensor_config = ExperimentConfig(**{**config.__dict__, "mujoco_sensor_bias_scale": 0.7})
    sensor_second = generate_experiment_stream(
        sensor_config,
        "sensor_bias",
        10,
        env_id="FakeEnv-v0",
    )
    actuator_first = generate_experiment_stream(config, "actuator_loss", 10, env_id="FakeEnv-v0")
    actuator_config = ExperimentConfig(**{**config.__dict__, "mujoco_actuator_loss_scale": 0.4})
    actuator_second = generate_experiment_stream(
        actuator_config,
        "actuator_loss",
        10,
        env_id="FakeEnv-v0",
    )
    fixed_goal_config = ExperimentConfig(
        **{**config.__dict__, "mujoco_freeze_task_goal": True}
    )
    fixed_goal = generate_experiment_stream(
        fixed_goal_config,
        "target",
        10,
        env_id="FakeEnv-v0",
    )
    sinusoidal_config = ExperimentConfig(
        **{**config.__dict__, "mujoco_policy_mode": "sinusoidal"}
    )
    sinusoidal = generate_experiment_stream(
        sinusoidal_config,
        "gravity",
        10,
        env_id="FakeEnv-v0",
    )
    dither_config = ExperimentConfig(
        **{**config.__dict__, "mujoco_policy_mode": "linear_sinusoidal_dither"}
    )
    dither = generate_experiment_stream(
        dither_config,
        "gravity",
        10,
        env_id="FakeEnv-v0",
    )
    burst_config = ExperimentConfig(
        **{**config.__dict__, "mujoco_policy_mode": "linear_probe_burst"}
    )
    burst = generate_experiment_stream(
        burst_config,
        "gravity",
        10,
        env_id="FakeEnv-v0",
    )
    chirp_config = ExperimentConfig(
        **{**config.__dict__, "mujoco_policy_mode": "chirp"}
    )
    chirp = generate_experiment_stream(
        chirp_config,
        "gravity",
        10,
        env_id="FakeEnv-v0",
    )
    pulse_config = ExperimentConfig(
        **{**config.__dict__, "mujoco_policy_mode": "pulse"}
    )
    pulse = generate_experiment_stream(
        pulse_config,
        "gravity",
        10,
        env_id="FakeEnv-v0",
    )

    assert len(generated) == 14
    assert float(first.z[0, 0]) != float(second.z[0, 0])
    assert float(target_first.z[0, 0]) != float(target_second.z[0, 0])
    assert float(sensor_first.z[0, 0]) != float(sensor_second.z[0, 0])
    assert float(actuator_first.z[0, 0]) != float(actuator_second.z[0, 0])
    assert float(target_first.z[0, 0]) != float(fixed_goal.z[0, 0])
    assert float(first.z[0, 0]) != float(dither.z[0, 0])
    assert float(first.z[0, 0]) != float(burst.z[0, 0])
    assert float(first.z[0, 0]) != float(sinusoidal.z[0, 0])
    assert float(first.z[0, 0]) != float(chirp.z[0, 0])
    assert float(first.z[0, 0]) != float(pulse.z[0, 0])
    filenames = {path.name for path in tmp_path.glob("*.npz")}
    assert any("_g1_5_" in name for name in filenames)
    assert any("_g2_" in name for name in filenames)
    assert any("_t0_12_" in name for name in filenames)
    assert any("_t0_25_" in name for name in filenames)
    assert any("_sb0_35_" in name for name in filenames)
    assert any("_sb0_7_" in name for name in filenames)
    assert any("_al0_55_" in name for name in filenames)
    assert any("_al0_4_" in name for name in filenames)
    assert any("_fixedgoal_t0_12_" in name for name in filenames)
    assert all("_fb16_" in name for name in filenames)
    assert all("_std0_1_" in name for name in filenames)
    assert all("_policy" in name for name in filenames)
    assert all("_knn3_" in name for name in filenames)
    assert any("_policylinear_actor_" in name for name in filenames)
    assert any("_policylinear_sinusoidal_dither_" in name for name in filenames)
    assert any("_policylinear_probe_burst_" in name for name in filenames)
    assert any("_policysinusoidal_" in name for name in filenames)
    assert any("_policychirp_" in name for name in filenames)
    assert any("_policypulse_" in name for name in filenames)


def test_mujoco_trace_cache_loads_legacy_policy_variant(tmp_path, monkeypatch):
    import cadet.evaluation as evaluation_module

    legacy = SimulatedRun(
        z=np.full((4, 1), 7.0),
        endogenous_delta=np.zeros((4, 1)),
        tau=2,
        feature_names=("legacy_probe",),
        scenario="mujoco_FakeEnv-v0_gravity",
    )
    save_trace(
        legacy,
        tmp_path,
        source="mujoco",
        env_id="FakeEnv-v0",
        scenario="gravity",
        seed=10,
        trace_variant="model_component_probe_fb64_g1.45",
    )

    def fail_if_generated(config):
        raise AssertionError("legacy trace should load without regenerating MuJoCo")

    monkeypatch.setattr(evaluation_module, "generate_mujoco_internal_stream", fail_if_generated)
    config = ExperimentConfig(
        source="mujoco",
        horizon=4,
        tau=2,
        load_traces=str(tmp_path),
        mujoco_env_ids=("FakeEnv-v0",),
        mujoco_feature_mode="model_component_probe",
        mujoco_frozen_buffer_size=64,
        mujoco_policy_mode="linear_actor",
        mujoco_policy_std=0.35,
    )

    loaded = generate_experiment_stream(config, "gravity", 10, env_id="FakeEnv-v0")
    np.testing.assert_allclose(loaded.z, legacy.z)
    assert loaded.feature_names == legacy.feature_names


def test_locomotion_probe_feature_names_are_fixed_width():
    component_names = mujoco_feature_names("model_component_probe")
    observation_names = mujoco_feature_names("locomotion_observation_probe")
    dynamics_names = mujoco_feature_names("locomotion_dynamics_probe")
    stability_names = mujoco_feature_names("locomotion_stability_probe")
    gravity_names = mujoco_feature_names("gravity_signature_probe")
    gravity_residual_names = mujoco_feature_names("gravity_residual_probe")
    gravity_compensation_names = mujoco_feature_names("gravity_compensation_probe")
    gravity_balance_names = mujoco_feature_names("gravity_balance_probe")
    walker_gravity_names = mujoco_feature_names("walker_gravity_response_probe")
    walker_excitation_names = mujoco_feature_names("walker_excitation_response_probe")
    walker_innovation_gain_names = mujoco_feature_names("walker_innovation_gain_probe")
    gait_names = mujoco_feature_names("gait_phase_response_probe")
    gait_load_names = mujoco_feature_names("gait_load_response_probe")
    kernel_support_names = mujoco_feature_names("kernel_support_response_probe")
    mechanical_names = mujoco_feature_names("mechanical_energy_probe")
    actuator_efficiency_names = mujoco_feature_names("actuator_efficiency_probe")

    assert len(component_names) == 34
    assert component_names[0] == "model_obs_resid_00"
    assert component_names[-1] == "model_qvel_resid_08"
    assert len(observation_names) == 61
    assert observation_names[0] == "locomotion_obs_00"
    assert observation_names[-1] == "locomotion_reward_ctrl"
    assert len(dynamics_names) == 52
    assert dynamics_names[0] == "locomotion_cfrc_norm_00"
    assert dynamics_names[-1] == "locomotion_reward_ctrl"
    assert len(stability_names) == 10
    assert stability_names[0] == "stability_done"
    assert stability_names[-1] == "stability_obs_delta_norm"
    assert len(gravity_names) == 26
    assert gravity_names[0] == "gravity_qfrc_bias_norm"
    assert gravity_names[-1] == "gravity_abs_height_delta_per_action"
    assert len(gravity_residual_names) == len(gravity_names)
    assert gravity_residual_names[0] == "gravity_qfrc_bias_norm_resid"
    assert gravity_residual_names[-1] == "gravity_abs_height_delta_per_action_resid"
    assert len(gravity_compensation_names) == 18
    assert gravity_compensation_names[0] == "gravity_comp_qfrc_bias_resid_00"
    assert gravity_compensation_names[-1] == "gravity_comp_bias_passive_alignment_resid"
    assert len(gravity_balance_names) == 18
    assert gravity_balance_names[0] == "gravity_balance_log_load"
    assert gravity_balance_names[-1] == "gravity_balance_load_concentration"
    assert len(walker_gravity_names) == 24
    assert walker_gravity_names[0] == "walker_gravity_bias_vertical_resid"
    assert walker_gravity_names[-1] == "walker_gravity_action_vertical_response_resid"
    assert len(walker_excitation_names) == 25
    assert walker_excitation_names[0] == "walker_excitation_root_forward_delta_resid"
    assert walker_excitation_names[-1] == "walker_excitation_abs_action_delta_joint_alignment_resid"
    assert len(walker_innovation_gain_names) == 25
    assert walker_innovation_gain_names[0] == "walker_gain_forward_sum_resid"
    assert walker_innovation_gain_names[-1] == "walker_gain_innovation_joint_alignment_resid"
    assert len(gait_names) == 24
    assert gait_names[0] == "gait_leg_velocity_resid_00"
    assert gait_names[-1] == "gait_pitch_rate_per_action_resid"
    assert len(gait_load_names) == 36
    assert gait_load_names[0] == "gait_load_total_resid"
    assert gait_load_names[-1] == "gait_loaded_speed_share_resid_07"
    assert len(kernel_support_names) == 36
    assert kernel_support_names[0] == "kernel_support_load_total_resid"
    assert kernel_support_names[-1] == "kernel_support_loaded_speed_share_resid_07"
    assert len(mechanical_names) == 18
    assert mechanical_names[0] == "mechanical_kinetic_proxy"
    assert mechanical_names[-1] == "mechanical_reward_per_action"
    assert len(actuator_efficiency_names) == 13
    assert actuator_efficiency_names[0] == "actuator_action_norm"
    assert actuator_efficiency_names[-1] == "actuator_angle_delta"


def test_locomotion_probe_vectors_are_finite_and_fixed_width():
    transition = _Transition(
        obs=np.linspace(-0.5, 0.5, 17),
        action=np.linspace(-0.2, 0.2, 6),
        reward=1.25,
        next_obs=np.linspace(-0.4, 0.6, 17),
        done=True,
        info={
            "x_velocity": 0.7,
            "reward_forward": 0.8,
            "reward_ctrl": -0.1,
        },
        qpos=np.linspace(0.0, 0.8, 9),
        qvel=np.linspace(-0.4, 0.4, 9),
        next_qpos=np.linspace(0.1, 0.9, 9),
        next_qvel=np.linspace(-0.3, 0.5, 9),
        physics=np.concatenate(
            [
                np.linspace(0.1, 0.8, 8),
                np.linspace(-1.0, 1.0, 9),
                np.linspace(0.1, 0.9, 9),
                np.linspace(-2.0, 2.0, 9),
            ]
        ),
    )

    class _Data:
        cfrc_ext = np.arange(48, dtype=float).reshape(8, 6)
        qfrc_constraint = np.linspace(-1.0, 1.0, 9)
        qfrc_passive = np.linspace(0.1, 0.9, 9)
        qfrc_bias = np.linspace(-0.5, 0.5, 9)
        qfrc_actuator = np.linspace(-0.7, 0.7, 9)
        qacc = np.linspace(-2.0, 2.0, 9)
        qvel = np.linspace(-0.3, 0.5, 9)
        xpos = np.column_stack(
            [
                np.linspace(0.0, 0.7, 8),
                np.linspace(0.1, 0.8, 8),
                np.linspace(0.15, 0.85, 8),
            ]
        )
        cvel = np.column_stack(
            [
                np.zeros((8, 3)),
                np.linspace(-0.4, 0.4, 24).reshape(8, 3),
            ]
        )
        ncon = 2
        contact = (
            SimpleNamespace(dist=-0.02),
            SimpleNamespace(dist=0.03),
        )

    class _Unwrapped:
        data = _Data()

    class _Env:
        unwrapped = _Unwrapped()

    vectors = {
        "motion_probe": _motion_probe_feature_vector(transition),
        "locomotion_observation_probe": _locomotion_observation_probe_feature_vector(
            transition
        ),
        "locomotion_dynamics_probe": _locomotion_dynamics_probe_feature_vector(
            transition,
            _Env(),
        ),
        "locomotion_stability_probe": _locomotion_stability_probe_feature_vector(
            transition,
            episode_step=37,
        ),
        "gravity_signature_probe": _gravity_signature_probe_feature_vector(
            transition,
            _Env(),
        ),
        "gravity_residual_probe": np.zeros(len(mujoco_feature_names("gravity_residual_probe"))),
        "gravity_compensation_probe": _gravity_compensation_probe_target_vector(
            transition,
            _Env(),
        ),
        "gravity_balance_probe": _gravity_balance_probe_feature_vector(transition, _Env()),
        "walker_gravity_response_probe": _walker_gravity_response_probe_target_vector(
            transition,
            _Env(),
        ),
        "walker_excitation_response_probe": _walker_excitation_response_probe_target_vector(
            transition
        ),
        "walker_innovation_gain_probe": _walker_innovation_gain_probe_target_vector(
            transition
        ),
        "gait_load_response_probe": _support_response_probe_target_vector(
            transition,
            _Env(),
        ),
        "mechanical_energy_probe": _mechanical_energy_probe_feature_vector(transition),
        "actuator_efficiency_probe": _actuator_efficiency_probe_feature_vector(transition),
        "contact_slip_probe": _contact_slip_probe_feature_vector(_Env()),
        "support_response_probe": _support_response_probe_target_vector(transition, _Env()),
        "kernel_support_response_probe": _support_response_probe_target_vector(
            transition,
            _Env(),
        ),
    }
    for mode, vector in vectors.items():
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (len(mujoco_feature_names(mode)),)
        assert np.all(np.isfinite(vector))

    physics = _physics_probe_target_vector(_Env())
    assert physics.shape == (len(mujoco_feature_names("physics_residual_probe")),)
    assert np.all(np.isfinite(physics))
    force_balance = _force_balance_probe_target_vector(_Env())
    assert force_balance.shape == (len(mujoco_feature_names("force_balance_probe")),)
    assert np.all(np.isfinite(force_balance))


def test_gait_phase_response_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("gait_phase_response_probe")
    transitions = []
    for idx in range(5):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.5, 0.5, 17) + 0.01 * idx,
                action=np.linspace(-0.4, 0.4, 6) + 0.02 * idx,
                reward=1.0 + 0.1 * idx,
                next_obs=np.linspace(-0.45, 0.55, 17) + 0.01 * idx,
                done=False,
                info={
                    "x_velocity": 0.7 + 0.02 * idx,
                    "reward_forward": 0.8,
                    "reward_ctrl": -0.1,
                },
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.05, 0.85, 9) + 0.02 * idx,
                next_qvel=np.linspace(-0.35, 0.45, 9) + 0.03 * idx,
            )
        )
    probe = _FrozenGaitPhaseResponseProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:4])
    vector = probe.feature_vector(transitions[4])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))


def test_gait_load_response_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("gait_load_response_probe")
    transitions = []
    for idx in range(5):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.5, 0.5, 17) + 0.01 * idx,
                action=np.linspace(-0.4, 0.4, 6) + 0.02 * idx,
                reward=1.0 + 0.1 * idx,
                next_obs=np.linspace(-0.45, 0.55, 17) + 0.01 * idx,
                done=False,
                info={
                    "x_velocity": 0.7 + 0.02 * idx,
                    "reward_forward": 0.8,
                    "reward_ctrl": -0.1,
                },
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.05, 0.85, 9) + 0.02 * idx,
                next_qvel=np.linspace(-0.35, 0.45, 9) + 0.03 * idx,
                physics=np.linspace(-1.0, 1.0, len(names)) + 0.05 * idx,
            )
        )
    probe = _FrozenGaitLoadResponseProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:4])
    vector = probe.feature_vector(transitions[4])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))


def test_kernel_support_response_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("kernel_support_response_probe")
    transitions = []
    for idx in range(5):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.5, 0.5, 17) + 0.01 * idx,
                action=np.linspace(-0.4, 0.4, 6) + 0.02 * idx,
                reward=1.0 + 0.1 * idx,
                next_obs=np.linspace(-0.45, 0.55, 17) + 0.01 * idx,
                done=False,
                info={
                    "x_velocity": 0.7 + 0.02 * idx,
                    "reward_forward": 0.8,
                    "reward_ctrl": -0.1,
                },
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.05, 0.85, 9) + 0.02 * idx,
                next_qvel=np.linspace(-0.35, 0.45, 9) + 0.03 * idx,
                physics=np.linspace(-1.0, 1.0, len(names)) + 0.05 * idx,
            )
        )
    probe = _FrozenKernelSupportResponseProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:4])
    vector = probe.feature_vector(transitions[4])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))
    np.testing.assert_allclose(
        probe._input_vector(transitions[0]),
        probe._input_vector(transitions[0]),
    )


def test_physics_residual_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("physics_residual_probe")
    transitions = []
    for idx in range(4):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.2, 0.2, 17),
                action=np.linspace(-0.3, 0.3, 6) + 0.01 * idx,
                reward=1.0,
                next_obs=np.linspace(-0.1, 0.3, 17),
                done=False,
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.1, 0.9, 9),
                next_qvel=np.linspace(-0.3, 0.5, 9),
                physics=np.linspace(-1.0, 1.0, len(names)) + 0.05 * idx,
            )
        )

    probe = _FrozenPhysicsResidualProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:3])
    vector = probe.feature_vector(transitions[3])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))


def test_force_balance_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("force_balance_probe")
    transitions = []
    for idx in range(4):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.2, 0.2, 17),
                action=np.linspace(-0.3, 0.3, 6) + 0.01 * idx,
                reward=1.0,
                next_obs=np.linspace(-0.1, 0.3, 17),
                done=False,
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.1, 0.9, 9),
                next_qvel=np.linspace(-0.3, 0.5, 9),
                physics=np.linspace(-1.0, 1.0, len(names)) + 0.05 * idx,
            )
        )

    probe = _FrozenForceBalanceResidualProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:3])
    vector = probe.feature_vector(transitions[3])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))


def test_gravity_residual_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("gravity_residual_probe")
    transitions = []
    for idx in range(4):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.2, 0.2, 17),
                action=np.linspace(-0.3, 0.3, 6) + 0.01 * idx,
                reward=1.0,
                next_obs=np.linspace(-0.1, 0.3, 17),
                done=False,
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.1, 0.9, 9),
                next_qvel=np.linspace(-0.3, 0.5, 9),
                physics=np.linspace(-1.0, 1.0, len(names)) + 0.05 * idx,
            )
        )

    probe = _FrozenGravityResidualProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:3])
    vector = probe.feature_vector(transitions[3])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))


def test_gravity_compensation_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("gravity_compensation_probe")
    transitions = []
    for idx in range(5):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.2, 0.2, 17),
                action=np.linspace(-0.3, 0.3, 6) + 0.01 * idx,
                reward=1.0,
                next_obs=np.linspace(-0.1, 0.3, 17),
                done=False,
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.1, 0.9, 9),
                next_qvel=np.linspace(-0.3, 0.5, 9),
                physics=np.linspace(-1.0, 1.0, len(names)) + 0.05 * idx,
            )
        )

    probe = _FrozenGravityCompensationProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:4])
    vector = probe.feature_vector(transitions[4])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))


def test_walker_gravity_response_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("walker_gravity_response_probe")
    transitions = []
    for idx in range(5):
        transitions.append(
            _Transition(
                obs=np.linspace(-0.2, 0.2, 17),
                action=np.linspace(-0.3, 0.3, 6) + 0.01 * idx,
                reward=1.0,
                next_obs=np.linspace(-0.1, 0.3, 17),
                done=False,
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.1, 0.9, 9) + 0.02 * idx,
                next_qvel=np.linspace(-0.3, 0.5, 9) + 0.03 * idx,
                physics=np.linspace(-1.0, 1.0, len(names)) + 0.05 * idx,
            )
        )

    probe = _FrozenWalkerGravityResponseProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:4])
    vector = probe.feature_vector(transitions[4])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))
    np.testing.assert_allclose(
        probe._input_vector(transitions[0]),
        probe._input_vector(transitions[0]),
    )


def test_walker_excitation_response_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("walker_excitation_response_probe")
    transitions = []
    previous_action = np.zeros(6, dtype=float)
    for idx in range(5):
        action = np.linspace(-0.3, 0.3, 6) + 0.01 * idx
        transitions.append(
            _Transition(
                obs=np.linspace(-0.2, 0.2, 17),
                action=action,
                reward=1.0,
                next_obs=np.linspace(-0.1, 0.3, 17),
                done=False,
                info={"previous_action": previous_action.copy()},
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.1, 0.9, 9) + 0.02 * idx,
                next_qvel=np.linspace(-0.3, 0.5, 9) + 0.03 * idx,
            )
        )
        previous_action = action

    target = _walker_excitation_response_probe_target_vector(transitions[-1])
    assert target.shape == (len(names),)
    assert np.all(np.isfinite(target))

    probe = _FrozenWalkerExcitationResponseProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:4])
    vector = probe.feature_vector(transitions[4])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))
    assert not np.allclose(
        probe._input_vector(transitions[1]),
        probe._input_vector(
            _Transition(
                obs=transitions[1].obs,
                action=transitions[1].action,
                reward=transitions[1].reward,
                next_obs=transitions[1].next_obs,
                done=transitions[1].done,
                info={"previous_action": np.ones(6, dtype=float)},
                qpos=transitions[1].qpos,
                qvel=transitions[1].qvel,
                next_qpos=transitions[1].next_qpos,
                next_qvel=transitions[1].next_qvel,
            )
        ),
    )


def test_walker_innovation_gain_probe_is_finite_and_fixed_width():
    names = mujoco_feature_names("walker_innovation_gain_probe")
    transitions = []
    previous_action = np.zeros(6, dtype=float)
    previous_innovation = np.zeros(6, dtype=float)
    for idx in range(5):
        action = np.linspace(-0.3, 0.3, 6) + 0.01 * idx
        policy_mean = np.linspace(-0.2, 0.2, 6) + 0.005 * idx
        policy_innovation = (action - policy_mean) / 0.05
        transitions.append(
            _Transition(
                obs=np.linspace(-0.2, 0.2, 17),
                action=action,
                reward=1.0,
                next_obs=np.linspace(-0.1, 0.3, 17),
                done=False,
                info={
                    "previous_action": previous_action.copy(),
                    "policy_mean": policy_mean.copy(),
                    "policy_std": 0.05,
                    "policy_innovation": policy_innovation.copy(),
                    "previous_policy_innovation": previous_innovation.copy(),
                },
                qpos=np.linspace(0.0, 0.8, 9) + 0.02 * idx,
                qvel=np.linspace(-0.4, 0.4, 9) + 0.03 * idx,
                next_qpos=np.linspace(0.1, 0.9, 9) + 0.02 * idx,
                next_qvel=np.linspace(-0.3, 0.5, 9) + 0.03 * idx,
            )
        )
        previous_action = action
        previous_innovation = policy_innovation

    target = _walker_innovation_gain_probe_target_vector(transitions[-1])
    assert target.shape == (len(names),)
    assert np.all(np.isfinite(target))

    probe = _FrozenWalkerInnovationGainProbe()
    assert probe.feature_vector(transitions[0]).shape == (len(names),)
    probe.fit(transitions[:4])
    vector = probe.feature_vector(transitions[4])
    assert vector.shape == (len(names),)
    assert np.all(np.isfinite(vector))
    reset_transition = _Transition(
        obs=transitions[1].obs,
        action=transitions[1].action,
        reward=transitions[1].reward,
        next_obs=transitions[1].next_obs,
        done=transitions[1].done,
        info={
            **dict(transitions[1].info or {}),
            "previous_policy_innovation": np.zeros(6, dtype=float),
        },
        qpos=transitions[1].qpos,
        qvel=transitions[1].qvel,
        next_qpos=transitions[1].next_qpos,
        next_qvel=transitions[1].next_qvel,
    )
    assert not np.allclose(
        probe._input_vector(transitions[1]),
        probe._input_vector(reset_transition),
    )


def test_detectability_audit_reports_metric_separations():
    run = generate_rl_internal_stream(horizon=260, tau=120, d=3, scenario="scale", seed=77)
    rows = detectability_rows_for_stream(
        run,
        scenario="scale",
        run_idx=0,
        seed=77,
        env_id="synthetic",
        window=80,
    )
    assert len(rows) == 3
    assert {"scale_separation", "dynamics_separation", "detectable"} <= set(rows[0])
    assert rows[0]["audit_status"] == "diagnostic_not_null_calibrated"
    assert rows[0]["interpretation"] == "residual_metric_separation_diagnostic_only"
    assert any(bool(row["detectable"]) for row in rows)


def test_experiment_suite_labels_detectability_audit_as_diagnostic(tmp_path):
    config = ExperimentConfig(
        horizon=90,
        tau=45,
        n_runs=1,
        eval_runs=1,
        d=2,
        scale_window=15,
        dynamics_window=18,
        warmup=25,
        anytime=False,
        seed=91,
    )

    run_experiment_suite(
        config,
        tmp_path,
        scenarios=["no_shift", "scale"],
        include_references=False,
    )

    audit = pd.read_csv(tmp_path / "detectability_audit.csv")

    assert not audit.empty
    assert set(audit["audit_status"]) == {"diagnostic_not_null_calibrated"}
    assert set(audit["interpretation"]) == {
        "residual_metric_separation_diagnostic_only",
    }


def test_exp002_writes_manifest_from_defined_config(tmp_path, monkeypatch):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp002_synthetic_baseline_calibration.py"
    )
    module = _load_script_module("exp002_synthetic_baseline_calibration", script_path)

    monkeypatch.setattr(
        module,
        "calibrate_baselines",
        lambda config, target_far, baselines, alpha_grid: (
            {name: 0.01 for name in module.DEFAULT_BASELINES},
            pd.DataFrame([{"detector": module.DEFAULT_BASELINES[0], "alpha": 0.01}]),
            pd.DataFrame(
                [
                    {
                        "detector": module.DEFAULT_BASELINES[0],
                        "calibrated_alpha": 0.01,
                        "target_far": target_far,
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "default_detector_factories",
        lambda config, alpha_by_detector, include_references: {
            name: (lambda name=name: None) for name in module.DEFAULT_BASELINES
        },
    )
    monkeypatch.setattr(
        module,
        "evaluate_suite",
        lambda config, scenarios, detector_factories, include_references: pd.DataFrame(
            [
                {
                    "scenario": "no_shift",
                    "detector": module.DEFAULT_BASELINES[0],
                    "false_alarm": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "summarize_results",
        lambda results: pd.DataFrame(
            [
                {
                    "scenario": "no_shift",
                    "detector": module.DEFAULT_BASELINES[0],
                    "false_alarm_rate": 0.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "write_summary_markdown",
        lambda summary, path: Path(path).write_text("summary\n", encoding="utf-8"),
    )

    settings = module.Experiment2Settings(outdir=tmp_path)
    module.run(settings)

    manifest = json.loads((tmp_path / "trace_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "synthetic"
    assert manifest["calibration_seed"] == module.CALIBRATION_SEED
    assert manifest["evaluation_seed"] == module.EVALUATION_SEED


def test_exp002_reuses_calibration_streams_across_alpha_grid(monkeypatch):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp002_synthetic_baseline_calibration.py"
    )
    module = _load_script_module("exp002_synthetic_baseline_calibration_cached", script_path)

    generated = 0

    def fake_generate(config, scenario, seed, env_id=None):
        nonlocal generated
        generated += 1
        return generate_rl_internal_stream(
            horizon=20,
            tau=None,
            d=1,
            scenario="no_shift",
            seed=seed,
        )

    monkeypatch.setattr(module, "generate_experiment_stream", fake_generate)
    monkeypatch.setattr(module, "_detector_factory", lambda *args, **kwargs: (lambda d: object()))
    monkeypatch.setattr(module, "run_detector", lambda detector, stream: None)

    settings = module.Experiment2Settings(runs=3)
    config = module.build_config(123, settings)
    calibrated, diagnostics, selected = module.calibrate_baselines(
        config,
        target_far=0.10,
        baselines=("A", "B"),
        alpha_grid=(1e-3, 1e-2, 1e-1),
    )

    assert generated == settings.runs
    assert set(calibrated) == {"A", "B"}
    assert diagnostics.shape[0] == 6
    assert selected.shape[0] == 2


def test_experiment_script_validation_rejects_bad_numeric_inputs():
    exp001_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp001_synthetic_matched_far.py"
    )
    exp002_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp002_synthetic_baseline_calibration.py"
    )
    exp003_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp003_empirical_baseline_calibration.py"
    )

    exp001 = _load_script_module("exp001_synthetic_matched_far", exp001_path)
    exp002 = _load_script_module("exp002_synthetic_baseline_calibration_validation", exp002_path)
    exp003 = _load_script_module("exp003_empirical_baseline_calibration_validation", exp003_path)

    with pytest.raises(ValueError, match="runs must be positive"):
        exp001.validate_settings(exp001.Experiment1Settings(runs=0))
    with pytest.raises(ValueError, match="alpha values"):
        exp002.validate_settings(exp002.Experiment2Settings(alpha_grid=(0.0, 0.1)))
    with pytest.raises(ValueError, match="warmup \\+ scale_window"):
        exp003.validate_settings(exp003.Experiment3Settings(horizon=170, tau=80))


def test_exp006_carries_probe_and_cadet_threshold_modes(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp006_mujoco_empirical_calibrated.py"
    )
    module = _load_script_module("exp006_mujoco_empirical_calibrated_modes", script_path)
    settings = module.Experiment6Settings(
        outdir=tmp_path,
        mujoco_envs=("HalfCheetah-v5",),
        scenarios=("no_shift", "gravity"),
        mujoco_feature_mode="actor_value_reset_probe",
        mujoco_probe_update_mode="frozen_policy",
        mujoco_policy_std=0.05,
        mujoco_policy_mode="sinusoidal",
        target_shift=0.2,
        sensor_bias_scale=0.4,
        actuator_loss_scale=0.6,
        freeze_task_goal=True,
        mujoco_frozen_buffer_size=32,
        mujoco_knn_neighbors=4,
        center_mode="block",
        cadet_threshold_mode="empirical_statistic",
        cadet_heldout_runs=3,
        allow_no_admissible_baselines=True,
    )
    module.validate_settings(settings)
    config = module.build_config(123, 2, settings)
    assert config.mujoco_feature_mode == "actor_value_reset_probe"
    assert config.mujoco_probe_update_mode == "frozen_policy"
    assert config.mujoco_policy_std == 0.05
    assert config.mujoco_policy_mode == "sinusoidal"
    assert config.mujoco_target_shift == 0.2
    assert config.mujoco_sensor_bias_scale == 0.4
    assert config.mujoco_actuator_loss_scale == 0.6
    assert config.mujoco_freeze_task_goal
    assert config.mujoco_frozen_buffer_size == 32
    assert config.mujoco_knn_neighbors == 4
    assert config.center_mode == "block"
    assert settings.cadet_threshold_mode == "empirical_statistic"
    assert settings.allow_no_admissible_baselines


def test_exp006_per_env_calibration_requires_empirical_cadet_mode(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp006_mujoco_empirical_calibrated.py"
    )
    module = _load_script_module("exp006_mujoco_empirical_calibrated_scope", script_path)

    module.validate_settings(
        module.Experiment6Settings(
            outdir=tmp_path,
            mujoco_envs=("HalfCheetah-v5", "Walker2d-v5"),
            scenarios=("no_shift", "gravity"),
            cadet_threshold_mode="empirical_statistic",
            calibration_scope="per_env",
        )
    )
    with pytest.raises(ValueError, match="per-env calibration"):
        module.validate_settings(
            module.Experiment6Settings(
                outdir=tmp_path,
                mujoco_envs=("HalfCheetah-v5", "Walker2d-v5"),
                scenarios=("no_shift", "gravity"),
                cadet_threshold_mode="matched_alpha",
                calibration_scope="per_env",
            )
        )


def test_exp006_accepts_component_locomotion_probe_modes(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp006_mujoco_empirical_calibrated.py"
    )
    module = _load_script_module("exp006_mujoco_empirical_calibrated_locomotion", script_path)

    for mode in (
        "model_component_probe",
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
        "actuator_efficiency_probe",
        "physics_residual_probe",
        "force_balance_probe",
        "contact_slip_probe",
        "support_telemetry_probe",
        "support_response_probe",
        "kernel_support_response_probe",
        "sensor_consistency_probe",
    ):
        settings = module.Experiment6Settings(
            outdir=tmp_path / mode,
            mujoco_envs=("HalfCheetah-v5",),
            scenarios=("no_shift", "gravity"),
            mujoco_feature_mode=mode,
            allow_no_admissible_baselines=True,
        )
        module.validate_settings(settings)
        assert module.build_config(123, 2, settings).mujoco_feature_mode == mode


def test_exp006_accepts_diagnostic_policy_modes(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp006_mujoco_empirical_calibrated.py"
    )
    module = _load_script_module("exp006_mujoco_empirical_calibrated_policies", script_path)

    for mode in (
        "linear_sinusoidal_dither",
        "linear_probe_burst",
        "zero",
        "sinusoidal",
        "chirp",
        "pulse",
    ):
        settings = module.Experiment6Settings(
            outdir=tmp_path / mode,
            mujoco_envs=("HalfCheetah-v5",),
            scenarios=("no_shift", "gravity"),
            mujoco_policy_mode=mode,
            allow_no_admissible_baselines=True,
        )
        module.validate_settings(settings)
        assert module.build_config(123, 2, settings).mujoco_policy_mode == mode


def test_exp007_channel_screened_trace_cache_contract(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "exp007_channel_screened_cadet.py"
    module = _load_script_module("exp007_channel_screened_cadet_contract", script_path)

    with pytest.raises(ValueError, match="one of --load-traces or --save-traces"):
        module.validate_settings(
            module.ScreenedReplaySettings(
                outdir=tmp_path / "out",
                load_traces=None,
                save_traces=None,
                mujoco_envs=("HalfCheetah-v5",),
                scenarios=("no_shift", "gravity"),
            )
        )

    with pytest.raises(ValueError, match="load-traces directory does not exist"):
        module.validate_settings(
            module.ScreenedReplaySettings(
                outdir=tmp_path / "out",
                load_traces=tmp_path / "missing",
                save_traces=None,
                mujoco_envs=("HalfCheetah-v5",),
                scenarios=("no_shift", "gravity"),
            )
        )

    settings = module.ScreenedReplaySettings(
        outdir=tmp_path / "out",
        load_traces=tmp_path / "missing",
        save_traces=tmp_path / "generated",
        mujoco_envs=("HalfCheetah-v5",),
        scenarios=("no_shift", "sensor_bias"),
        sensor_bias_scale=0.45,
        actuator_loss_scale=0.65,
        mujoco_probe_update_mode="frozen_policy",
        arms=("anchored_location",),
        dense_cauchy=True,
        dense_cauchy_fraction=0.25,
        screened_baselines=("Moment",),
    )
    module.validate_settings(settings)
    config = module.build_config(settings, 123, 2)
    assert config.load_traces == str(tmp_path / "missing")
    assert config.save_traces == str(tmp_path / "generated")
    assert config.mujoco_sensor_bias_scale == 0.45
    assert config.mujoco_actuator_loss_scale == 0.65
    assert config.mujoco_probe_update_mode == "frozen_policy"
    detector = module.cadet_detector(
        config,
        settings.arms,
        dense_cauchy=settings.dense_cauchy,
        dense_cauchy_fraction=settings.dense_cauchy_fraction,
    )
    assert detector.config.arms == ("anchored_location",)
    assert detector.config.dense_cauchy
    assert detector.config.dense_cauchy_fraction == pytest.approx(0.25)
    assert settings.screened_baselines == ("Moment",)
    assert module.channel_name_allowed(
        "locomotion_qacc_01",
        include_regex=("locomotion_qacc",),
        exclude_regex=("constraint",),
    )
    assert not module.channel_name_allowed(
        "locomotion_qfrc_constraint_01",
        include_regex=("locomotion_",),
        exclude_regex=("constraint",),
    )
    with pytest.raises(ValueError, match="invalid channel-screen regex"):
        module.validate_settings(
            module.ScreenedReplaySettings(
                outdir=tmp_path / "out",
                load_traces=tmp_path / "missing",
                save_traces=tmp_path / "generated",
                mujoco_envs=("HalfCheetah-v5",),
                scenarios=("no_shift", "gravity"),
                screen_include_regex=("[",),
            )
        )
    with pytest.raises(ValueError, match="dense-cauchy-fraction"):
        module.validate_settings(
            module.ScreenedReplaySettings(
                outdir=tmp_path / "out",
                load_traces=tmp_path / "missing",
                save_traces=tmp_path / "generated",
                mujoco_envs=("HalfCheetah-v5",),
                scenarios=("no_shift", "gravity"),
                dense_cauchy_fraction=1.0,
            )
        )


def test_exp007_channel_screened_baseline_replay_smoke(tmp_path, monkeypatch):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "exp007_channel_screened_cadet.py"
    module = _load_script_module("exp007_channel_screened_cadet_baseline_smoke", script_path)

    def fake_generate(config, scenario, seed, env_id=None):
        rng = np.random.default_rng(seed)
        z = 0.01 * rng.normal(size=(config.horizon, 2))
        tau = None
        if scenario != "no_shift":
            tau = config.tau
            z[int(config.tau) :, 0] += 3.0
        return SimulatedRun(
            z=z,
            endogenous_delta=np.zeros_like(z),
            tau=tau,
            feature_names=("probe_a", "probe_b"),
            scenario=f"fake_{scenario}",
        )

    monkeypatch.setattr(module, "generate_experiment_stream", fake_generate)
    settings = module.ScreenedReplaySettings(
        outdir=tmp_path / "out",
        load_traces=None,
        save_traces=tmp_path / "traces",
        mujoco_envs=("FakeEnv-v0",),
        scenarios=("no_shift", "gravity"),
        target_far=0.5,
        horizon=60,
        tau=30,
        calibration_runs=2,
        heldout_runs=2,
        eval_runs=2,
        scale_window=5,
        dynamics_window=8,
        warmup=10,
        screen_max_statistic=10.0,
        screened_baselines=("Moment",),
    )

    raw, summary = module.run(settings)

    assert {"CADET_screened", "Moment_screened"} <= set(raw["detector"])
    assert {"CADET_screened", "Moment_screened"} <= set(summary["detector"])
    thresholds = pd.read_csv(settings.outdir / "screened_empirical_thresholds.csv")
    heldout = pd.read_csv(settings.outdir / "screened_heldout_far.csv")
    assert {"CADET_screened", "Moment_screened"} <= set(thresholds["detector"])
    assert {"CADET_screened", "Moment_screened"} <= set(heldout["detector"])


def test_cadet_detects_simulated_scale_change_with_pur():
    run = generate_rl_internal_stream(
        horizon=650,
        tau=300,
        scenario="scale",
        seed=13,
        endogenous_step=0.05,
    )
    detector = CADETDetector(
        CADETConfig(
            alpha=0.5,
            scale_window=60,
            dynamics_window=80,
            warmup=120,
            anytime=False,
            arms=("scale",),
        ),
        residualizer=LinearPURResidualizer(),
    )
    alarm = detector.run(run.z, endogenous_delta=run.endogenous_delta)
    assert alarm is not None
    assert alarm.time >= run.tau
    assert alarm.time - run.tau < 180


def test_cadet_reports_candidate_statistics_for_empirical_calibration():
    detector = CADETDetector(
        CADETConfig(
            alpha=1e-300,
            scale_window=10,
            dynamics_window=12,
            warmup=20,
            anytime=False,
            arms=("scale", "dynamics", "location"),
        ),
        residualizer=IdentityResidualizer(),
    )
    last_result = None
    for value in np.sin(np.linspace(0.0, 8.0, 60)):
        last_result = detector.update(np.asarray([value]), endogenous_delta=np.asarray([0.0]))
    assert last_result is not None
    assert last_result.statistics
    assert {candidate.arm for candidate in last_result.statistics} <= {
        "scale",
        "dynamics",
        "location",
    }
    assert any(candidate.arm == "location" for candidate in last_result.statistics)


# --- Revision additions -------------------------------------------------------


def test_clopper_pearson_upper_bound_matches_known_values():
    from cadet.calibration import clopper_pearson_upper_bound

    # 0/20 no-shift alarms is compatible with a true rate near 0.14, far above the
    # nominal alpha = 0.05 target; 0/50 still only certifies 0.058.
    assert clopper_pearson_upper_bound(0, 20) == pytest.approx(0.1391, abs=1e-3)
    assert clopper_pearson_upper_bound(0, 50) == pytest.approx(0.0582, abs=1e-3)
    assert clopper_pearson_upper_bound(5, 5) == 1.0
    assert np.isnan(clopper_pearson_upper_bound(0, 0))
    with pytest.raises(ValueError):
        clopper_pearson_upper_bound(3, 2)


def test_residualized_detector_feeds_baseline_the_pur_residual():
    from cadet.baselines import BaselineConfig, MomentMonitor, ResidualizedDetector

    rng = np.random.default_rng(7)
    z = rng.normal(size=(50, 3))
    delta = np.full((50, 3), 0.1)
    wrapped = ResidualizedDetector(MomentMonitor(BaselineConfig(warmup=10, window=10)))
    seen: list[np.ndarray] = []
    inner_update = wrapped.inner.update

    def spy(row, endogenous_delta=None):
        seen.append(np.asarray(row, dtype=float).copy())
        return inner_update(row, endogenous_delta=endogenous_delta)

    wrapped.inner.update = spy
    for idx in range(z.shape[0]):
        wrapped.update(z[idx], endogenous_delta=delta[idx])

    expected = z - np.cumsum(delta, axis=0)
    assert np.allclose(np.asarray(seen), expected)
    assert wrapped.name == "Moment_PUR"


def test_pur_baselines_are_registered_and_wrap_the_right_detector():
    from cadet.baselines import KernelMMDDetector, ResidualizedDetector
    from cadet.evaluation import ExperimentConfig, default_detector_factories

    factories = default_detector_factories(ExperimentConfig())
    for name in ("MMD_PUR", "ADWIN_PUR", "CUSUM_PUR", "BOCPD_PUR", "Energy_PUR"):
        assert name in factories
    detector = factories["MMD_PUR"](5)
    assert isinstance(detector, ResidualizedDetector)
    assert isinstance(detector.inner, KernelMMDDetector)


def test_dense_scenario_moves_every_channel():
    from cadet.simulation import generate_rl_internal_stream

    d = 6
    run = generate_rl_internal_stream(horizon=4000, tau=2000, d=d, scenario="dense", seed=5)
    residual = run.z - np.cumsum(run.endogenous_delta, axis=0)
    ratio = residual[2000:].std(axis=0) / residual[:2000].std(axis=0)
    # Dense alternative: every channel moves, none dominates.
    assert np.all(ratio > 1.1)
    assert ratio.max() / ratio.min() < 1.5

    concentrated = generate_rl_internal_stream(
        horizon=4000, tau=2000, d=d, scenario="scale", seed=5
    )
    c_residual = concentrated.z - np.cumsum(concentrated.endogenous_delta, axis=0)
    c_ratio = c_residual[2000:].std(axis=0) / c_residual[:2000].std(axis=0)
    assert c_ratio.max() / c_ratio.min() > 2.0


def test_lambda_phi_monitor_does_not_flag_independent_data():
    """The monitor re-tests every lag on every channel on every block.

    With a per-lag two-sigma cut-off at least one spurious exceedance is near-certain on
    independent data, which made the first version flag almost every block. The cut-off
    is corrected for the number of simultaneous tests; this pins that behaviour.
    """

    from cadet.detectors import CADETConfig, CADETDetector

    rng = np.random.default_rng(19)
    flagged = 0
    for _ in range(10):
        detector = CADETDetector(
            CADETConfig(scale_window=20, dynamics_window=40, warmup=80,
                        lambda_phi=1.0, lambda_phi_monitor_block=100)
        )
        for row in rng.normal(size=(900, 5)):
            detector.update(row, endogenous_delta=np.zeros(5))
        flagged += int(detector.lambda_phi_status == "calibration_invalid")
    assert flagged == 0


def test_multiplicity_corrected_z_is_stricter_than_two_sigma():
    from cadet.utils import estimate_lambda_phi, multiplicity_corrected_z

    assert multiplicity_corrected_z(1) == pytest.approx(1.96, abs=0.01)
    assert multiplicity_corrected_z(250) > 3.5
    # A stricter cut-off can only discard more lags, so the estimate cannot increase.
    rng = np.random.default_rng(4)
    x = rng.normal(size=400)
    assert estimate_lambda_phi(x, noise_floor_z=multiplicity_corrected_z(250)) <= (
        estimate_lambda_phi(x, noise_floor_z=2.0)
    )


def test_lambda_phi_monitor_flags_non_conservative_constant():
    from cadet.detectors import CADETConfig, CADETDetector

    rng = np.random.default_rng(3)
    strongly_dependent = np.zeros((600, 3), dtype=float)
    for t in range(1, 600):
        strongly_dependent[t] = 0.9 * strongly_dependent[t - 1] + rng.normal(size=3)

    config = CADETConfig(
        scale_window=20, dynamics_window=40, warmup=80, lambda_phi=1.0,
        lambda_phi_monitor_block=100,
    )
    detector = CADETDetector(config)
    for row in strongly_dependent:
        detector.update(row, endogenous_delta=np.zeros(3))
    assert detector.lambda_phi_status == "calibration_invalid"
    assert detector.lambda_phi_monitor["max_estimate"] > 1.0

    independent = rng.normal(size=(600, 3))
    generous = CADETDetector(
        CADETConfig(
            scale_window=20, dynamics_window=40, warmup=80, lambda_phi=4.0,
            lambda_phi_monitor_block=100,
        )
    )
    for row in independent:
        generous.update(row, endogenous_delta=np.zeros(3))
    assert generous.lambda_phi_status == "conservative"


def test_lambda_phi_monitor_is_off_by_default():
    from cadet.detectors import CADETConfig, CADETDetector

    detector = CADETDetector(CADETConfig(scale_window=10, dynamics_window=20, warmup=20))
    rng = np.random.default_rng(1)
    for row in rng.normal(size=(100, 2)):
        detector.update(row, endogenous_delta=np.zeros(2))
    assert detector.lambda_phi_status == "not_monitored"


def test_pur_update_diagnostics_reports_first_order_remainder():
    from cadet.mujoco import _LinearActorValue, _pur_update_diagnostics

    pytest.importorskip("gymnasium")
    from cadet.mujoco import _Transition

    rng = np.random.default_rng(0)
    obs_dim, action_dim = 5, 2
    before = _LinearActorValue(
        obs_dim, action_dim, 8, -np.ones(action_dim), np.ones(action_dim), 0.2, 0
    )
    buffer = [
        _Transition(
            obs=rng.normal(size=obs_dim),
            action=rng.normal(size=action_dim) * 0.1,
            reward=float(rng.normal()),
            next_obs=rng.normal(size=obs_dim),
            done=False,
        )
        for _ in range(12)
    ]
    after = before.copy()
    after.update(buffer[0], 0.99, 2e-4, 1e-3)

    row = _pur_update_diagnostics(before, after, buffer, 0.99, 1e-3, step=10, tau=5)
    assert row["post_change"] is True
    assert row["delta_omega_norm"] > 0.0
    # The first-order JVP must track the exact frozen-buffer increment closely for a
    # single clipped actor-critic step; the remainder is the Lemma 2 O(||dw||^2) term.
    assert row["remainder_norm"] <= 0.25 * max(row["exact_norm"], 1e-12)


def test_cauchy_combination_ignores_saturated_p_values():
    from cadet.utils import cauchy_combination_p_value

    # The arms report conservative bounds that equal exactly 1 when the statistic is
    # small. tan((0.5-p)*pi) diverges there, so an unrestricted combination is dominated
    # by non-informative arms and returns 1 regardless of the evidence.
    assert cauchy_combination_p_value([1.0] * 5) == 1.0
    assert cauchy_combination_p_value([1.0, 1.0, 1.0, 1.0, 0.02]) == pytest.approx(0.02, abs=1e-6)
    assert cauchy_combination_p_value([0.3] * 5) == pytest.approx(0.3, abs=1e-6)
    # A single informative arm among many saturated ones must still register.
    assert cauchy_combination_p_value([0.01] + [1.0] * 20) < 0.05
    assert cauchy_combination_p_value([]) == 1.0


def test_scenario_scale_override_controls_channel_loading():
    from cadet.simulation import generate_rl_internal_stream

    d = 6
    override = np.full(d, 1.6)
    run = generate_rl_internal_stream(
        horizon=4000, tau=2000, d=d, scenario="scale", seed=11, post_scale_override=override
    )
    residual = run.z - np.cumsum(run.endogenous_delta, axis=0)
    ratio = residual[2000:].std(axis=0) / residual[:2000].std(axis=0)
    assert np.all(ratio > 1.2)

    with pytest.raises(ValueError):
        generate_rl_internal_stream(
            horizon=100, tau=50, d=d, scenario="scale", seed=1,
            post_scale_override=np.ones(d + 1),
        )


def test_scale_arm_statistic_is_lambda_phi_invariant():
    """Lambda_phi must enter the scale arm's threshold, never its statistic.

    Matched-FAR calibration refits the threshold on the run-level statistic, so a
    statistic that moved with Lambda_phi would make the mixing constant silently change
    the deployed detections. The dynamics arm deliberately reports a threshold-normalised
    statistic, so its value does scale with Lambda_phi; that asymmetry is pinned here.
    """

    from cadet.detectors import DynamicsArm, ScaleArm

    rng = np.random.default_rng(23)
    values = rng.normal(size=400)

    statistics = []
    for lam in (1.0, 3.0, 10.0):
        arm = ScaleArm(window=20, lambda_phi=lam)
        best = 0.0
        for value in values:
            result = arm.update(float(value), 0.05)
            if result.ready:
                best = max(best, float(result.statistic))
        statistics.append(best)
    assert len(set(round(v, 12) for v in statistics)) == 1

    dynamics = []
    for lam in (1.0, 10.0):
        arm = DynamicsArm(window=40, pilot_size=80, lambda_phi=lam)
        best = 0.0
        for value in values:
            result = arm.update(float(value), 0.05)
            if result.ready:
                best = max(best, float(result.statistic))
        dynamics.append(best)
    assert dynamics[0] > dynamics[1] > 0.0
