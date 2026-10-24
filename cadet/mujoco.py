"""MuJoCo benchmark trace generation for CADET.

The module is optional: importing it does not require Gymnasium/MuJoCo, but
calling ``generate_mujoco_internal_stream`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from cadet.simulation import DEFAULT_FEATURE_NAMES, SimulatedRun


MujocoShift = Literal[
    "no_shift",
    "gravity",
    "friction",
    "target",
    "sensor_bias",
    "actuator_loss",
]
MujocoPolicyMode = Literal[
    "linear_actor",
    "linear_sinusoidal_dither",
    "linear_probe_burst",
    "zero",
    "sinusoidal",
    "chirp",
    "pulse",
]
MujocoFeatureMode = Literal[
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
]
MujocoProbeUpdateMode = Literal["online", "frozen_policy"]

KINEMATIC_FEATURE_NAMES = (
    "reward",
    "obs_norm",
    "obs_delta_norm",
    "action_norm",
    "qpos_rel_norm",
    "qvel_norm",
)
TELEMETRY_FEATURE_NAMES = (
    "x_velocity",
    "x_position",
    "reward_forward",
    "reward_ctrl",
    "reward_survive",
    "z_distance_from_origin",
)
DOMAIN_FEATURE_NAMES = (
    "x_velocity",
    "reward_forward",
    "reward_ctrl",
    "reward_survive",
    "z_distance_from_origin",
)
RESET_PROBE_FEATURE_NAMES = (
    "probe_reward_delta",
    "probe_obs_delta_norm",
    "probe_qpos_delta_norm",
    "probe_qvel_delta_norm",
)
MODEL_PROBE_FEATURE_NAMES = (
    "model_reward_abs_error",
    "model_obs_delta_error_norm",
    "model_qpos_delta_error_norm",
    "model_qvel_delta_error_norm",
)
MODEL_STATE_PROBE_FEATURE_NAMES = MODEL_PROBE_FEATURE_NAMES[1:]
MODEL_VECTOR_PROBE_FEATURE_NAMES = (
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
MODEL_COMPONENT_PROBE_FEATURE_NAMES = (
    tuple(f"model_obs_resid_{idx:02d}" for idx in range(17))
    + tuple(f"model_qpos_rel_resid_{idx:02d}" for idx in range(8))
    + tuple(f"model_qvel_resid_{idx:02d}" for idx in range(9))
)
MOTION_PROBE_FEATURE_NAMES = (
    "motion_obs_delta_norm",
    "motion_qpos_delta_rel_norm",
    "motion_qvel_delta_norm",
    "motion_qvel_delta_max_abs",
)
LOCOMOTION_OBSERVATION_PROBE_FEATURE_NAMES = (
    tuple(f"locomotion_obs_{idx:02d}" for idx in range(17))
    + tuple(f"locomotion_obs_delta_{idx:02d}" for idx in range(17))
    + tuple(f"locomotion_qpos_rel_{idx:02d}" for idx in range(8))
    + tuple(f"locomotion_qvel_{idx:02d}" for idx in range(9))
    + tuple(f"locomotion_action_{idx:02d}" for idx in range(6))
    + (
        "locomotion_reward",
        "locomotion_x_velocity",
        "locomotion_reward_forward",
        "locomotion_reward_ctrl",
    )
)
LOCOMOTION_DYNAMICS_PROBE_FEATURE_NAMES = (
    tuple(f"locomotion_cfrc_norm_{idx:02d}" for idx in range(8))
    + tuple(f"locomotion_qfrc_constraint_{idx:02d}" for idx in range(9))
    + tuple(f"locomotion_qfrc_passive_{idx:02d}" for idx in range(9))
    + tuple(f"locomotion_qacc_{idx:02d}" for idx in range(9))
    + tuple(f"locomotion_qvel_{idx:02d}" for idx in range(9))
    + (
        "locomotion_contact_norm",
        "locomotion_constraint_norm",
        "locomotion_passive_norm",
        "locomotion_qacc_norm",
        "locomotion_reward",
        "locomotion_x_velocity",
        "locomotion_reward_forward",
        "locomotion_reward_ctrl",
    )
)
PHYSICS_RESIDUAL_PROBE_FEATURE_NAMES = (
    tuple(f"physics_cfrc_norm_resid_{idx:02d}" for idx in range(8))
    + tuple(f"physics_qfrc_constraint_resid_{idx:02d}" for idx in range(9))
    + tuple(f"physics_qfrc_passive_resid_{idx:02d}" for idx in range(9))
    + tuple(f"physics_qacc_resid_{idx:02d}" for idx in range(9))
)
FORCE_BALANCE_PROBE_FEATURE_NAMES = (
    tuple(f"force_cfrc_ext_resid_{idx:02d}" for idx in range(48))
    + tuple(f"force_qfrc_constraint_resid_{idx:02d}" for idx in range(9))
    + tuple(f"force_qfrc_passive_resid_{idx:02d}" for idx in range(9))
    + tuple(f"force_qfrc_bias_resid_{idx:02d}" for idx in range(9))
    + tuple(f"force_qfrc_actuator_resid_{idx:02d}" for idx in range(9))
    + tuple(f"force_qacc_resid_{idx:02d}" for idx in range(9))
)
CONTACT_SLIP_PROBE_FEATURE_NAMES = (
    (
        "contact_count",
        "contact_dist_min",
        "contact_dist_mean",
        "contact_dist_max",
        "contact_load_total",
        "contact_loaded_speed_total",
    )
    + tuple(f"contact_body_z_{idx:02d}" for idx in range(8))
    + tuple(f"contact_body_xy_speed_{idx:02d}" for idx in range(8))
    + tuple(f"contact_body_vertical_speed_{idx:02d}" for idx in range(8))
    + tuple(f"contact_body_load_norm_{idx:02d}" for idx in range(8))
    + tuple(f"contact_loaded_xy_speed_{idx:02d}" for idx in range(8))
    + tuple(f"contact_low_height_xy_speed_{idx:02d}" for idx in range(8))
)
SUPPORT_RESPONSE_PROBE_FEATURE_NAMES = (
    (
        "support_load_total_resid",
        "support_load_per_action_resid",
        "support_load_per_qacc_resid",
        "support_loaded_speed_total_resid",
        "support_loaded_speed_per_load_resid",
        "support_low_height_speed_total_resid",
        "support_contact_count_resid",
        "support_penetration_mean_resid",
        "support_penetration_max_resid",
        "support_load_entropy_resid",
        "support_load_concentration_resid",
        "support_qacc_norm_resid",
        "support_actuated_delta_norm_resid",
        "support_vertical_delta_resid",
        "support_height_delta_resid",
        "support_pitch_delta_resid",
        "support_bias_per_load_resid",
        "support_passive_per_load_resid",
        "support_constraint_per_load_resid",
        "support_bias_qacc_alignment_resid",
    )
    + tuple(f"support_load_share_resid_{idx:02d}" for idx in range(8))
    + tuple(f"support_loaded_speed_share_resid_{idx:02d}" for idx in range(8))
)
SUPPORT_TELEMETRY_PROBE_FEATURE_NAMES = tuple(
    name.replace("_resid", "") for name in SUPPORT_RESPONSE_PROBE_FEATURE_NAMES
)
KERNEL_SUPPORT_RESPONSE_PROBE_FEATURE_NAMES = tuple(
    f"kernel_{name}" for name in SUPPORT_RESPONSE_PROBE_FEATURE_NAMES
)
LOCOMOTION_STABILITY_PROBE_FEATURE_NAMES = (
    "stability_done",
    "stability_episode_age",
    "stability_reward",
    "stability_x_velocity",
    "stability_reward_forward",
    "stability_reward_ctrl",
    "stability_qpos_height",
    "stability_qpos_angle",
    "stability_qvel_norm",
    "stability_obs_delta_norm",
)
GRAVITY_SIGNATURE_PROBE_FEATURE_NAMES = (
    "gravity_qfrc_bias_norm",
    "gravity_qfrc_bias_vertical",
    "gravity_qfrc_bias_pitch",
    "gravity_qfrc_passive_norm",
    "gravity_qfrc_constraint_norm",
    "gravity_qfrc_actuator_norm",
    "gravity_qacc_norm",
    "gravity_vertical_accel",
    "gravity_pitch_accel",
    "gravity_height",
    "gravity_height_delta",
    "gravity_vertical_velocity",
    "gravity_body_angle",
    "gravity_body_angle_delta",
    "gravity_contact_load_total",
    "gravity_contact_load_per_height",
    "gravity_contact_load_per_qacc",
    "gravity_bias_per_load",
    "gravity_constraint_per_load",
    "gravity_passive_per_load",
    "gravity_bias_qacc_alignment",
    "gravity_actuator_qacc_alignment",
    "gravity_vertical_balance_proxy",
    "gravity_vertical_bias_per_accel",
    "gravity_low_height_load",
    "gravity_abs_height_delta_per_action",
)
GRAVITY_RESIDUAL_PROBE_FEATURE_NAMES = tuple(
    f"{name}_resid" for name in GRAVITY_SIGNATURE_PROBE_FEATURE_NAMES
)
GRAVITY_COMPENSATION_PROBE_FEATURE_NAMES = (
    tuple(f"gravity_comp_qfrc_bias_resid_{idx:02d}" for idx in range(9))
    + (
        "gravity_comp_bias_norm_resid",
        "gravity_comp_bias_vertical_resid",
        "gravity_comp_bias_pitch_resid",
        "gravity_comp_bias_root_norm_resid",
        "gravity_comp_bias_actuated_norm_resid",
        "gravity_comp_bias_per_height_resid",
        "gravity_comp_vertical_per_height_resid",
        "gravity_comp_bias_qacc_alignment_resid",
        "gravity_comp_bias_passive_alignment_resid",
    )
)
GRAVITY_BALANCE_PROBE_FEATURE_NAMES = (
    "gravity_balance_log_load",
    "gravity_balance_log_load_per_height",
    "gravity_balance_log_load_per_qacc",
    "gravity_balance_bias_vertical_per_load",
    "gravity_balance_passive_vertical_per_load",
    "gravity_balance_constraint_vertical_per_load",
    "gravity_balance_actuator_vertical_per_load",
    "gravity_balance_bias_vertical_per_accel",
    "gravity_balance_total_vertical_per_load",
    "gravity_balance_total_vertical_per_accel",
    "gravity_balance_bias_qacc_alignment",
    "gravity_balance_actuator_qacc_alignment",
    "gravity_balance_height_delta_per_action",
    "gravity_balance_vertical_velocity",
    "gravity_balance_pitch_delta_per_action",
    "gravity_balance_low_height_load",
    "gravity_balance_load_entropy",
    "gravity_balance_load_concentration",
)
WALKER_GRAVITY_RESPONSE_PROBE_FEATURE_NAMES = (
    "walker_gravity_bias_vertical_resid",
    "walker_gravity_bias_pitch_resid",
    "walker_gravity_bias_root_norm_resid",
    "walker_gravity_bias_actuated_norm_resid",
    "walker_gravity_bias_per_height_resid",
    "walker_gravity_bias_per_load_resid",
    "walker_gravity_vertical_qacc_resid",
    "walker_gravity_pitch_qacc_resid",
    "walker_gravity_total_vertical_balance_resid",
    "walker_gravity_vertical_balance_per_load_resid",
    "walker_gravity_height_delta_resid",
    "walker_gravity_pitch_delta_resid",
    "walker_gravity_vertical_velocity_resid",
    "walker_gravity_pitch_rate_resid",
    "walker_gravity_load_total_resid",
    "walker_gravity_load_per_height_resid",
    "walker_gravity_low_height_load_resid",
    "walker_gravity_load_concentration_resid",
    "walker_gravity_load_entropy_resid",
    "walker_gravity_loaded_vertical_speed_resid",
    "walker_gravity_low_height_vertical_speed_resid",
    "walker_gravity_bias_qacc_alignment_resid",
    "walker_gravity_bias_passive_alignment_resid",
    "walker_gravity_action_vertical_response_resid",
)
WALKER_EXCITATION_RESPONSE_PROBE_FEATURE_NAMES = (
    "walker_excitation_root_forward_delta_resid",
    "walker_excitation_root_vertical_delta_resid",
    "walker_excitation_root_pitch_delta_resid",
    "walker_excitation_height_delta_resid",
    "walker_excitation_pitch_delta_resid",
    "walker_excitation_forward_per_innovation_resid",
    "walker_excitation_vertical_per_innovation_resid",
    "walker_excitation_pitch_per_innovation_resid",
    "walker_excitation_height_per_innovation_resid",
    "walker_excitation_pitch_angle_per_innovation_resid",
) + tuple(
    f"walker_excitation_joint_delta_resid_{idx:02d}" for idx in range(6)
) + tuple(
    f"walker_excitation_joint_per_innovation_resid_{idx:02d}" for idx in range(6)
) + (
    "walker_excitation_leg_delta_asym_resid",
    "walker_excitation_action_delta_joint_alignment_resid",
    "walker_excitation_abs_action_delta_joint_alignment_resid",
)
WALKER_INNOVATION_GAIN_PROBE_FEATURE_NAMES = (
    "walker_gain_forward_sum_resid",
    "walker_gain_vertical_sum_resid",
    "walker_gain_pitch_asym_resid",
    "walker_gain_height_norm_resid",
    "walker_gain_pitch_angle_norm_resid",
) + tuple(
    f"walker_gain_joint_cross_resid_{idx:02d}" for idx in range(6)
) + tuple(
    f"walker_gain_pitch_phase_resid_{idx:02d}" for idx in range(6)
) + tuple(
    f"walker_gain_vertical_phase_resid_{idx:02d}" for idx in range(6)
) + (
    "walker_gain_leg_delta_asym_resid",
    "walker_gain_innovation_joint_alignment_resid",
)
GAIT_PHASE_RESPONSE_PROBE_FEATURE_NAMES = (
    tuple(f"gait_leg_velocity_resid_{idx:02d}" for idx in range(6))
    + tuple(f"gait_joint_accel_resid_{idx:02d}" for idx in range(6))
    + (
        "gait_height_delta_resid",
        "gait_pitch_delta_resid",
        "gait_forward_velocity_resid",
        "gait_vertical_velocity_resid",
        "gait_pitch_rate_resid",
        "gait_leg_angle_asym_resid",
        "gait_leg_velocity_asym_resid",
        "gait_action_velocity_alignment_resid",
        "gait_abs_action_velocity_alignment_resid",
        "gait_forward_per_action_resid",
        "gait_vertical_per_action_resid",
        "gait_pitch_rate_per_action_resid",
    )
)
GAIT_LOAD_RESPONSE_PROBE_FEATURE_NAMES = (
    (
        "gait_load_total_resid",
        "gait_load_per_action_resid",
        "gait_load_per_qacc_resid",
        "gait_loaded_speed_total_resid",
        "gait_loaded_speed_per_load_resid",
        "gait_low_height_speed_total_resid",
        "gait_contact_count_resid",
        "gait_penetration_mean_resid",
        "gait_penetration_max_resid",
        "gait_load_entropy_resid",
        "gait_load_concentration_resid",
        "gait_qacc_norm_resid",
        "gait_actuated_delta_norm_resid",
        "gait_vertical_delta_resid",
        "gait_height_delta_resid",
        "gait_pitch_delta_resid",
        "gait_bias_per_load_resid",
        "gait_passive_per_load_resid",
        "gait_constraint_per_load_resid",
        "gait_bias_qacc_alignment_resid",
    )
    + tuple(f"gait_load_share_resid_{idx:02d}" for idx in range(8))
    + tuple(f"gait_loaded_speed_share_resid_{idx:02d}" for idx in range(8))
)
MECHANICAL_ENERGY_PROBE_FEATURE_NAMES = (
    "mechanical_kinetic_proxy",
    "mechanical_kinetic_delta",
    "mechanical_actuated_kinetic_proxy",
    "mechanical_actuated_kinetic_delta",
    "mechanical_height",
    "mechanical_height_delta",
    "mechanical_body_angle",
    "mechanical_body_angle_delta",
    "mechanical_action_norm",
    "mechanical_action_power",
    "mechanical_action_accel_alignment",
    "mechanical_passive_power",
    "mechanical_constraint_power",
    "mechanical_qacc_norm",
    "mechanical_forward_velocity",
    "mechanical_forward_per_action",
    "mechanical_ctrl_reward_per_action",
    "mechanical_reward_per_action",
)
CONTROL_RESPONSE_PROBE_FEATURE_NAMES = (
    tuple(f"control_qvel_delta_resid_{idx:02d}" for idx in range(6))
    + tuple(f"control_action_response_resid_{idx:02d}" for idx in range(6))
    + tuple(f"control_damping_work_resid_{idx:02d}" for idx in range(6))
    + tuple(f"control_response_ratio_resid_{idx:02d}" for idx in range(6))
    + (
        "control_height_delta_resid",
        "control_body_angle_delta_resid",
        "control_x_velocity_resid",
        "control_reward_forward_resid",
        "control_reward_ctrl_resid",
        "control_reward_resid",
        "control_action_norm_resid",
        "control_actuated_qvel_norm_resid",
        "control_qvel_delta_norm_resid",
        "control_forward_per_action_resid",
    )
)
ACTUATOR_EFFICIENCY_PROBE_FEATURE_NAMES = (
    "actuator_action_norm",
    "actuator_actuated_qvel_norm",
    "actuator_qvel_delta_norm",
    "actuator_delta_per_action",
    "actuator_projected_delta_per_action",
    "actuator_abs_projected_delta_per_action",
    "actuator_alignment",
    "actuator_x_velocity_per_action",
    "actuator_forward_reward_per_action",
    "actuator_ctrl_reward_per_action",
    "actuator_reward_per_action",
    "actuator_height_delta",
    "actuator_angle_delta",
)
SENSOR_CONSISTENCY_PROBE_FEATURE_NAMES = tuple(
    f"sensor_obs_resid_{idx:02d}" for idx in range(17)
)
TASK_REWARD_PROBE_FEATURE_NAMES = (
    "task_reward",
    "task_reward_dist",
    "task_reward_near",
    "task_reward_forward",
)
TASK_OBSERVATION_PROBE_FEATURE_NAMES = (
    "task_obs_coord_0",
    "task_obs_coord_1",
    "task_obs_coord_norm",
    "task_obs_coord_delta_norm",
)
KNN_STATE_PROBE_FEATURE_NAMES = (
    "knn_obs_delta_error_norm",
    "knn_qpos_delta_rel_error_norm",
    "knn_qvel_delta_error_norm",
)
PHASE_STATE_PROBE_FEATURE_NAMES = (
    "phase_obs_delta_error_norm",
    "phase_qpos_delta_rel_error_norm",
    "phase_qvel_delta_error_norm",
)


def mujoco_feature_names(mode: MujocoFeatureMode = "actor_value") -> tuple[str, ...]:
    """Return feature names emitted by a MuJoCo probe mode."""

    if mode == "actor_value":
        return DEFAULT_FEATURE_NAMES
    if mode == "actor_value_kinematic":
        return DEFAULT_FEATURE_NAMES + KINEMATIC_FEATURE_NAMES
    if mode == "actor_value_telemetry":
        return DEFAULT_FEATURE_NAMES + KINEMATIC_FEATURE_NAMES + TELEMETRY_FEATURE_NAMES
    if mode == "actor_value_domain":
        return DEFAULT_FEATURE_NAMES + KINEMATIC_FEATURE_NAMES + DOMAIN_FEATURE_NAMES
    if mode == "actor_value_reset_probe":
        return (
            DEFAULT_FEATURE_NAMES
            + KINEMATIC_FEATURE_NAMES
            + DOMAIN_FEATURE_NAMES
            + RESET_PROBE_FEATURE_NAMES
        )
    if mode == "reset_probe":
        return RESET_PROBE_FEATURE_NAMES
    if mode == "model_probe":
        return MODEL_PROBE_FEATURE_NAMES
    if mode == "model_state_probe":
        return MODEL_STATE_PROBE_FEATURE_NAMES
    if mode == "model_vector_probe":
        return MODEL_VECTOR_PROBE_FEATURE_NAMES
    if mode == "model_component_probe":
        return MODEL_COMPONENT_PROBE_FEATURE_NAMES
    if mode == "motion_probe":
        return MOTION_PROBE_FEATURE_NAMES
    if mode == "locomotion_observation_probe":
        return LOCOMOTION_OBSERVATION_PROBE_FEATURE_NAMES
    if mode == "locomotion_dynamics_probe":
        return LOCOMOTION_DYNAMICS_PROBE_FEATURE_NAMES
    if mode == "locomotion_stability_probe":
        return LOCOMOTION_STABILITY_PROBE_FEATURE_NAMES
    if mode == "gravity_signature_probe":
        return GRAVITY_SIGNATURE_PROBE_FEATURE_NAMES
    if mode == "gravity_residual_probe":
        return GRAVITY_RESIDUAL_PROBE_FEATURE_NAMES
    if mode == "gravity_compensation_probe":
        return GRAVITY_COMPENSATION_PROBE_FEATURE_NAMES
    if mode == "gravity_balance_probe":
        return GRAVITY_BALANCE_PROBE_FEATURE_NAMES
    if mode == "walker_gravity_response_probe":
        return WALKER_GRAVITY_RESPONSE_PROBE_FEATURE_NAMES
    if mode == "walker_excitation_response_probe":
        return WALKER_EXCITATION_RESPONSE_PROBE_FEATURE_NAMES
    if mode == "walker_innovation_gain_probe":
        return WALKER_INNOVATION_GAIN_PROBE_FEATURE_NAMES
    if mode == "gait_phase_response_probe":
        return GAIT_PHASE_RESPONSE_PROBE_FEATURE_NAMES
    if mode == "gait_load_response_probe":
        return GAIT_LOAD_RESPONSE_PROBE_FEATURE_NAMES
    if mode == "mechanical_energy_probe":
        return MECHANICAL_ENERGY_PROBE_FEATURE_NAMES
    if mode == "control_response_probe":
        return CONTROL_RESPONSE_PROBE_FEATURE_NAMES
    if mode == "actuator_efficiency_probe":
        return ACTUATOR_EFFICIENCY_PROBE_FEATURE_NAMES
    if mode == "sensor_consistency_probe":
        return SENSOR_CONSISTENCY_PROBE_FEATURE_NAMES
    if mode == "physics_residual_probe":
        return PHYSICS_RESIDUAL_PROBE_FEATURE_NAMES
    if mode == "force_balance_probe":
        return FORCE_BALANCE_PROBE_FEATURE_NAMES
    if mode == "contact_slip_probe":
        return CONTACT_SLIP_PROBE_FEATURE_NAMES
    if mode == "support_telemetry_probe":
        return SUPPORT_TELEMETRY_PROBE_FEATURE_NAMES
    if mode == "support_response_probe":
        return SUPPORT_RESPONSE_PROBE_FEATURE_NAMES
    if mode == "kernel_support_response_probe":
        return KERNEL_SUPPORT_RESPONSE_PROBE_FEATURE_NAMES
    if mode == "task_reward_probe":
        return TASK_REWARD_PROBE_FEATURE_NAMES
    if mode == "task_observation_probe":
        return TASK_OBSERVATION_PROBE_FEATURE_NAMES
    if mode == "knn_state_probe":
        return KNN_STATE_PROBE_FEATURE_NAMES
    if mode == "phase_state_probe":
        return PHASE_STATE_PROBE_FEATURE_NAMES
    raise ValueError(f"unknown MuJoCo feature mode {mode!r}")


@dataclass
class MujocoTraceConfig:
    env_id: str = "HalfCheetah-v5"
    horizon: int = 1200
    tau: int | None = 600
    shift: MujocoShift = "gravity"
    seed: int = 12345
    gamma: float = 0.99
    hidden_dim: int = 32
    policy_lr: float = 2e-4
    value_lr: float = 1e-3
    policy_std: float = 0.35
    policy_mode: MujocoPolicyMode = "linear_actor"
    gravity_scale: float = 1.45
    friction_scale: float = 1.8
    target_shift: float = 0.12
    sensor_bias_scale: float = 0.35
    actuator_loss_scale: float = 0.55
    freeze_task_goal: bool = False
    feature_mode: MujocoFeatureMode = "actor_value"
    probe_update_mode: MujocoProbeUpdateMode = "online"
    frozen_buffer_size: int = 96
    frozen_refresh: int | None = None
    knn_neighbors: int = 8
    # Opt-in PUR numerical diagnostics (Lemma 2 / post-change bias audit).
    # Off by default: enabling it does not change z_t, endogenous_delta or any
    # RNG draw, so cached traces stay byte-identical.
    record_update_diagnostics: bool = False
    jvp_probe_eps: float = 1e-3

    def __post_init__(self) -> None:
        mujoco_feature_names(self.feature_mode)
        if self.shift not in {
            "no_shift",
            "gravity",
            "friction",
            "target",
            "sensor_bias",
            "actuator_loss",
        }:
            raise ValueError(f"unknown MuJoCo shift {self.shift!r}")
        if self.probe_update_mode not in {"online", "frozen_policy"}:
            raise ValueError(f"unknown MuJoCo probe update mode {self.probe_update_mode!r}")
        if self.policy_mode not in {
            "linear_actor",
            "linear_sinusoidal_dither",
            "linear_probe_burst",
            "zero",
            "sinusoidal",
            "chirp",
            "pulse",
        }:
            raise ValueError(f"unknown MuJoCo policy mode {self.policy_mode!r}")
        if self.sensor_bias_scale <= 0.0:
            raise ValueError(f"sensor_bias_scale must be positive, got {self.sensor_bias_scale}")
        if not 0.0 < self.actuator_loss_scale <= 1.0:
            raise ValueError(
                "actuator_loss_scale must be in (0, 1], "
                f"got {self.actuator_loss_scale}"
            )
        if self.knn_neighbors <= 0:
            raise ValueError(f"knn_neighbors must be positive, got {self.knn_neighbors}")


@dataclass
class _Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    info: dict[str, Any] | None = None
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    next_qpos: np.ndarray | None = None
    next_qvel: np.ndarray | None = None
    physics: np.ndarray | None = None


def _pur_update_diagnostics(
    before: "_LinearActorValue",
    after: "_LinearActorValue",
    frozen_buffer: list["_Transition"],
    gamma: float,
    eps: float,
    *,
    step: int,
    tau: int | None,
) -> dict[str, float]:
    """Measure the quantities Lemma 2 and the post-change bias lemma assume.

    ``exact`` is the frozen-buffer increment the implementation actually subtracts,
    ``g_B(omega_t; M0) - g_B(omega_{t-1}; M0)``. ``jvp`` is the first-order
    Jacobian-vector product ``J_t Delta omega_t`` of eq. (7), obtained as a central
    directional derivative along ``Delta omega_t``. Their difference is the Taylor
    remainder ``R_t`` that Lemma 2 bounds by ``0.5 L_z ||Delta omega_t||^2``.
    """

    omega_before = before.trainable_vector()
    delta = after.trainable_vector() - omega_before
    delta_norm = float(np.linalg.norm(delta))

    exact = np.asarray(after.mean_features(frozen_buffer, gamma), dtype=float) - np.asarray(
        before.mean_features(frozen_buffer, gamma), dtype=float
    )
    if delta_norm <= 0.0:
        jvp = np.zeros_like(exact)
    else:
        plus = before.with_trainable_offset(delta, eps).mean_features(frozen_buffer, gamma)
        minus = before.with_trainable_offset(delta, -eps).mean_features(frozen_buffer, gamma)
        jvp = (np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)) / (2.0 * eps)

    remainder = exact - jvp
    exact_norm = float(np.linalg.norm(exact))
    remainder_norm = float(np.linalg.norm(remainder))
    return {
        "step": int(step),
        "post_change": bool(tau is not None and step >= int(tau)),
        "delta_omega_norm": delta_norm,
        "exact_norm": exact_norm,
        "jvp_norm": float(np.linalg.norm(jvp)),
        "remainder_norm": remainder_norm,
        "relative_remainder": remainder_norm / exact_norm if exact_norm > 1e-12 else 0.0,
        # Lemma 2 predicts remainder_norm <= 0.5 * L_z * delta_omega_norm**2; the
        # implied curvature constant makes that testable without knowing L_z.
        "implied_curvature": (
            2.0 * remainder_norm / (delta_norm**2) if delta_norm > 1e-12 else 0.0
        ),
    }


class _LinearActorValue:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        policy_std: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.hidden_w = rng.normal(scale=0.15 / np.sqrt(max(obs_dim, 1)), size=(obs_dim, hidden_dim))
        self.hidden_b = np.zeros(hidden_dim, dtype=float)
        self.policy_w = rng.normal(scale=0.08 / np.sqrt(max(hidden_dim, 1)), size=(hidden_dim, action_dim))
        self.policy_b = np.zeros(action_dim, dtype=float)
        self.value_w = rng.normal(scale=0.08 / np.sqrt(max(hidden_dim, 1)), size=hidden_dim)
        self.value_b = 0.0
        self.action_low = np.asarray(action_low, dtype=float)
        self.action_high = np.asarray(action_high, dtype=float)
        self.policy_std = float(policy_std)
        self.rng = rng

    def copy(self) -> "_LinearActorValue":
        clone = object.__new__(_LinearActorValue)
        clone.hidden_w = self.hidden_w.copy()
        clone.hidden_b = self.hidden_b.copy()
        clone.policy_w = self.policy_w.copy()
        clone.policy_b = self.policy_b.copy()
        clone.value_w = self.value_w.copy()
        clone.value_b = float(self.value_b)
        clone.action_low = self.action_low.copy()
        clone.action_high = self.action_high.copy()
        clone.policy_std = self.policy_std
        clone.rng = self.rng
        return clone

    # ---- PUR diagnostics -------------------------------------------------
    # ``hidden_w``/``hidden_b`` are a fixed random-feature layer and are never
    # updated, so the trainable parameter vector omega_t is exactly the
    # (policy_w, policy_b, value_w, value_b) block. Lemma 2's per-step cap
    # ||Delta omega_t|| <= eta G is checked numerically against this vector.

    def trainable_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.policy_w.reshape(-1),
                self.policy_b.reshape(-1),
                self.value_w.reshape(-1),
                np.asarray([self.value_b], dtype=float),
            ]
        )

    def with_trainable_offset(self, delta: np.ndarray, step: float = 1.0) -> "_LinearActorValue":
        """Return a copy with ``omega + step * delta`` in the trainable block."""

        clone = self.copy()
        flat = np.asarray(delta, dtype=float).reshape(-1)
        n_pw = self.policy_w.size
        n_pb = self.policy_b.size
        n_vw = self.value_w.size
        offset = 0
        clone.policy_w = self.policy_w + step * flat[offset : offset + n_pw].reshape(self.policy_w.shape)
        offset += n_pw
        clone.policy_b = self.policy_b + step * flat[offset : offset + n_pb].reshape(self.policy_b.shape)
        offset += n_pb
        clone.value_w = self.value_w + step * flat[offset : offset + n_vw].reshape(self.value_w.shape)
        offset += n_vw
        clone.value_b = float(self.value_b + step * flat[offset])
        return clone

    def _hidden(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        return np.tanh(obs @ self.hidden_w + self.hidden_b)

    def policy_mean(self, obs: np.ndarray) -> np.ndarray:
        hidden = self._hidden(obs)
        raw = hidden @ self.policy_w + self.policy_b
        return np.clip(raw, self.action_low, self.action_high)

    def value(self, obs: np.ndarray) -> float:
        hidden = self._hidden(obs)
        return float(hidden @ self.value_w + self.value_b)

    def act(self, obs: np.ndarray) -> np.ndarray:
        mean = self.policy_mean(obs)
        action = mean + self.policy_std * self.rng.normal(size=mean.shape)
        return np.clip(action, self.action_low, self.action_high)

    def bounded_entropy_proxy(self, mean: np.ndarray) -> float:
        """Approximate action entropy after clipping to finite action bounds."""

        action_range = np.maximum(self.action_high - self.action_low, 1e-8)
        lower_margin = mean - self.action_low
        upper_margin = self.action_high - mean
        relative_margin = np.clip(
            2.0 * np.minimum(lower_margin, upper_margin) / action_range,
            1e-6,
            1.0,
        )
        base_entropy = 0.5 * np.log(2.0 * np.pi * np.e * self.policy_std**2)
        return float(np.sum(base_entropy + np.log(relative_margin)))

    def feature_vector(self, transition: _Transition, gamma: float) -> np.ndarray:
        obs = transition.obs
        next_obs = transition.next_obs
        value = self.value(obs)
        next_value = 0.0 if transition.done else self.value(next_obs)
        td = transition.reward + gamma * next_value - value
        mean = self.policy_mean(obs)
        hidden = self._hidden(obs)
        entropy = self.bounded_entropy_proxy(mean)
        return np.asarray(
            [
                abs(td),
                value,
                float(entropy),
                float(np.max(mean)),
                float(np.linalg.norm(hidden)),
            ],
            dtype=float,
        )

    def mean_features(self, transitions: list[_Transition], gamma: float) -> np.ndarray:
        if not transitions:
            return np.zeros(len(DEFAULT_FEATURE_NAMES), dtype=float)
        return np.mean([self.feature_vector(tr, gamma) for tr in transitions], axis=0)

    def update(self, transition: _Transition, gamma: float, policy_lr: float, value_lr: float) -> None:
        obs = transition.obs
        hidden = self._hidden(obs)
        mean = self.policy_mean(obs)
        next_value = 0.0 if transition.done else self.value(transition.next_obs)
        value = self.value(obs)
        td = float(np.clip(transition.reward + gamma * next_value - value, -10.0, 10.0))
        advantage = td

        self.value_w += value_lr * td * hidden
        self.value_b += value_lr * td

        policy_error = (transition.action - mean) / max(self.policy_std**2, 1e-8)
        self.policy_w += policy_lr * advantage * np.outer(hidden, policy_error)
        self.policy_b += policy_lr * advantage * policy_error
        self.policy_w = np.clip(self.policy_w, -5.0, 5.0)
        self.policy_b = np.clip(self.policy_b, -5.0, 5.0)


class _FrozenLinearTransitionProbe:
    """Frozen ridge transition probe trained on early observed transitions."""

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = float(ridge)
        self.reward_beta: np.ndarray | None = None
        self.obs_delta_beta: np.ndarray | None = None
        self.qpos_delta_beta: np.ndarray | None = None
        self.qvel_delta_beta: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.reward_beta is not None and self.obs_delta_beta is not None

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(transition.obs, dtype=float).reshape(-1),
                np.asarray(transition.action, dtype=float).reshape(-1),
                np.ones(1, dtype=float),
            ]
        )

    def _fit_target(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gram = x.T @ x
        regularizer = self.ridge * np.eye(gram.shape[0], dtype=float)
        try:
            return np.linalg.solve(gram + regularizer, x.T @ y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram + regularizer, x.T @ y, rcond=None)[0]

    def fit(self, transitions: list[_Transition]) -> None:
        if not transitions:
            return
        x = np.vstack([self._input_vector(transition) for transition in transitions])
        rewards = np.asarray([transition.reward for transition in transitions], dtype=float)[:, None]
        obs_deltas = np.vstack(
            [
                np.asarray(transition.next_obs, dtype=float)
                - np.asarray(transition.obs, dtype=float)
                for transition in transitions
            ]
        )
        self.reward_beta = self._fit_target(x, rewards)
        self.obs_delta_beta = self._fit_target(x, obs_deltas)

        if all(transition.qpos is not None and transition.next_qpos is not None for transition in transitions):
            qpos_deltas = np.vstack(
                [
                    np.asarray(transition.next_qpos, dtype=float)
                    - np.asarray(transition.qpos, dtype=float)
                    for transition in transitions
                ]
            )
            self.qpos_delta_beta = self._fit_target(x, qpos_deltas)
        if all(transition.qvel is not None and transition.next_qvel is not None for transition in transitions):
            qvel_deltas = np.vstack(
                [
                    np.asarray(transition.next_qvel, dtype=float)
                    - np.asarray(transition.qvel, dtype=float)
                    for transition in transitions
                ]
            )
            self.qvel_delta_beta = self._fit_target(x, qvel_deltas)

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(MODEL_PROBE_FEATURE_NAMES), dtype=float)
        assert self.reward_beta is not None
        obs_residual, qpos_residual, qvel_residual = self.residual_vectors(transition)
        x = self._input_vector(transition)
        reward_pred = float((x @ self.reward_beta).reshape(-1)[0])
        return np.asarray(
            [
                abs(reward_pred - transition.reward),
                float(np.linalg.norm(obs_residual)),
                float(np.linalg.norm(qpos_residual)),
                float(np.linalg.norm(qvel_residual)),
            ],
            dtype=float,
        )

    def residual_vectors(
        self,
        transition: _Transition,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            return (
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )
        assert self.obs_delta_beta is not None
        x = self._input_vector(transition)
        obs_delta = np.asarray(transition.next_obs, dtype=float) - np.asarray(
            transition.obs,
            dtype=float,
        )
        obs_residual = (x @ self.obs_delta_beta) - obs_delta
        qpos_residual = np.zeros(0, dtype=float)
        if (
            self.qpos_delta_beta is not None
            and transition.qpos is not None
            and transition.next_qpos is not None
        ):
            qpos_delta = np.asarray(transition.next_qpos, dtype=float) - np.asarray(
                transition.qpos,
                dtype=float,
            )
            qpos_residual = (x @ self.qpos_delta_beta) - qpos_delta
        qvel_residual = np.zeros(0, dtype=float)
        if (
            self.qvel_delta_beta is not None
            and transition.qvel is not None
            and transition.next_qvel is not None
        ):
            qvel_delta = np.asarray(transition.next_qvel, dtype=float) - np.asarray(
                transition.qvel,
                dtype=float,
            )
            qvel_residual = (x @ self.qvel_delta_beta) - qvel_delta
        return obs_residual, qpos_residual, qvel_residual

    @staticmethod
    def signed_residual_projections(residual: np.ndarray) -> np.ndarray:
        values = np.asarray(residual, dtype=float).reshape(-1)
        if values.size == 0:
            return np.zeros(3, dtype=float)
        scale = np.sqrt(float(values.size))
        alternating = np.where(np.arange(values.size) % 2 == 0, 1.0, -1.0)
        ramp = np.linspace(-1.0, 1.0, values.size)
        return np.asarray(
            [
                float(np.sum(values) / scale),
                float(np.dot(alternating, values) / scale),
                float(np.dot(ramp, values) / scale),
            ],
            dtype=float,
        )

    def signed_projection_features(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(MODEL_VECTOR_PROBE_FEATURE_NAMES), dtype=float)
        residuals = self.residual_vectors(transition)
        return np.concatenate([self.signed_residual_projections(values) for values in residuals])

    def component_features(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(MODEL_COMPONENT_PROBE_FEATURE_NAMES), dtype=float)
        obs_residual, qpos_residual, qvel_residual = self.residual_vectors(transition)
        qpos_rel = qpos_residual[1:] if qpos_residual.size > 1 else qpos_residual
        return np.concatenate(
            [
                _fixed_width(obs_residual, 17),
                _fixed_width(qpos_rel, 8),
                _fixed_width(qvel_residual, 9),
            ]
        )


class _FrozenKNNTransitionProbe:
    """Frozen local transition probe trained on early observed transitions."""

    def __init__(self, neighbors: int = 8) -> None:
        self.neighbors = int(neighbors)
        self.x: np.ndarray | None = None
        self.x_center: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.obs_delta: np.ndarray | None = None
        self.qpos_delta_rel: np.ndarray | None = None
        self.qvel_delta: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.x is not None and self.obs_delta is not None

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(transition.obs, dtype=float).reshape(-1),
                np.asarray(transition.action, dtype=float).reshape(-1),
            ]
        )

    @staticmethod
    def _qpos_delta_rel(transition: _Transition) -> np.ndarray:
        if transition.qpos is None or transition.next_qpos is None:
            return np.zeros(0, dtype=float)
        delta = np.asarray(transition.next_qpos, dtype=float) - np.asarray(
            transition.qpos,
            dtype=float,
        )
        return delta[1:] if delta.size > 1 else delta

    @staticmethod
    def _qvel_delta(transition: _Transition) -> np.ndarray:
        if transition.qvel is None or transition.next_qvel is None:
            return np.zeros(0, dtype=float)
        return np.asarray(transition.next_qvel, dtype=float) - np.asarray(
            transition.qvel,
            dtype=float,
        )

    def fit(self, transitions: list[_Transition]) -> None:
        if not transitions:
            return
        x_raw = np.vstack([self._input_vector(transition) for transition in transitions])
        self.x_center = np.mean(x_raw, axis=0)
        self.x_scale = np.std(x_raw, axis=0) + 1e-6
        self.x = (x_raw - self.x_center) / self.x_scale
        self.obs_delta = np.vstack(
            [
                np.asarray(transition.next_obs, dtype=float)
                - np.asarray(transition.obs, dtype=float)
                for transition in transitions
            ]
        )
        self.qpos_delta_rel = np.vstack(
            [self._qpos_delta_rel(transition) for transition in transitions]
        )
        self.qvel_delta = np.vstack([self._qvel_delta(transition) for transition in transitions])

    def _predict(self, x_raw: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert self.x is not None
        assert self.x_center is not None
        assert self.x_scale is not None
        x = (x_raw - self.x_center) / self.x_scale
        distances = np.linalg.norm(self.x - x, axis=1)
        k = min(self.neighbors, len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        return np.mean(target[nearest], axis=0)

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(KNN_STATE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.obs_delta is not None
        assert self.qpos_delta_rel is not None
        assert self.qvel_delta is not None
        x = self._input_vector(transition)
        obs_delta = np.asarray(transition.next_obs, dtype=float) - np.asarray(
            transition.obs,
            dtype=float,
        )
        qpos_delta_rel = self._qpos_delta_rel(transition)
        qvel_delta = self._qvel_delta(transition)
        return np.asarray(
            [
                float(np.linalg.norm(self._predict(x, self.obs_delta) - obs_delta)),
                float(np.linalg.norm(self._predict(x, self.qpos_delta_rel) - qpos_delta_rel)),
                float(np.linalg.norm(self._predict(x, self.qvel_delta) - qvel_delta)),
            ],
            dtype=float,
        )


class _FrozenPhaseTransitionProbe(_FrozenKNNTransitionProbe):
    """Frozen local transition probe matched by phase-like locomotion state."""

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        qpos_rel = np.zeros(0, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = qpos[1:] if qpos.size > 1 else qpos
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        return np.concatenate(
            [
                qpos_rel,
                qvel,
                np.asarray(transition.action, dtype=float).reshape(-1),
            ]
        )


class _FrozenPhysicsResidualProbe:
    """Frozen ridge model for live force/acceleration telemetry residuals."""

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = float(ridge)
        self.beta: np.ndarray | None = None
        self.residual_center: np.ndarray | None = None
        self.residual_scale: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.beta is not None

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        qpos_rel = np.zeros(0, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = qpos[1:] if qpos.size > 1 else qpos
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        return np.concatenate(
            [
                qpos_rel,
                qvel,
                np.asarray(transition.action, dtype=float).reshape(-1),
                np.ones(1, dtype=float),
            ]
        )

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(PHYSICS_RESIDUAL_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(PHYSICS_RESIDUAL_PROBE_FEATURE_NAMES),
        )

    def fit(self, transitions: list[_Transition]) -> None:
        usable = [transition for transition in transitions if transition.physics is not None]
        if not usable:
            return
        x = np.vstack([self._input_vector(transition) for transition in usable])
        y = np.vstack([self._target_vector(transition) for transition in usable])
        gram = x.T @ x
        regularizer = self.ridge * np.eye(gram.shape[0], dtype=float)
        try:
            self.beta = np.linalg.solve(gram + regularizer, x.T @ y)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.lstsq(gram + regularizer, x.T @ y, rcond=None)[0]
        residuals = x @ self.beta - y
        self.residual_center = np.median(residuals, axis=0)
        q75, q25 = np.percentile(residuals, [75.0, 25.0], axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std_scale = np.std(residuals, axis=0)
        target_scale = np.std(y, axis=0)
        self.residual_scale = np.maximum.reduce(
            [
                np.asarray(iqr_scale, dtype=float),
                np.asarray(std_scale, dtype=float),
                0.1 * np.asarray(target_scale, dtype=float),
                np.full(y.shape[1], 1e-3, dtype=float),
            ]
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(PHYSICS_RESIDUAL_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(PHYSICS_RESIDUAL_PROBE_FEATURE_NAMES))


class _FrozenForceBalanceResidualProbe(_FrozenPhysicsResidualProbe):
    """Frozen ridge model for signed contact and generalized-force residuals."""

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        qpos_rel = np.zeros(0, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = qpos[1:] if qpos.size > 1 else qpos
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        actuated_qvel = _actuated_qvel(qvel, action.size)
        return np.concatenate(
            [
                qpos_rel,
                qvel,
                action,
                np.abs(action),
                action * actuated_qvel,
                np.ones(1, dtype=float),
            ]
        )

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(FORCE_BALANCE_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(FORCE_BALANCE_PROBE_FEATURE_NAMES),
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(FORCE_BALANCE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(FORCE_BALANCE_PROBE_FEATURE_NAMES))


class _FrozenSupportResponseProbe(_FrozenForceBalanceResidualProbe):
    """Frozen model for normalized support-load and motion-response residuals."""

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(SUPPORT_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(SUPPORT_RESPONSE_PROBE_FEATURE_NAMES),
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(SUPPORT_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(SUPPORT_RESPONSE_PROBE_FEATURE_NAMES))


class _FrozenKernelSupportResponseProbe(_FrozenSupportResponseProbe):
    """Frozen nonlinear state/action model for support-response residuals."""

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        qpos_rel = np.zeros(8, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = _fixed_width(qpos[1:] if qpos.size > 1 else qpos, 8)
        qvel = (
            _fixed_width(np.asarray(transition.qvel, dtype=float).reshape(-1), 9)
            if transition.qvel is not None
            else np.zeros(9, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        actuated_qvel = _actuated_qvel(qvel, action.size)
        height = float(qpos_rel[0]) if qpos_rel.size else 0.0
        pitch = float(qpos_rel[1]) if qpos_rel.size > 1 else 0.0
        base = np.concatenate(
            [
                qpos_rel,
                np.sin(qpos_rel),
                np.cos(qpos_rel),
                qvel,
                np.abs(qvel),
                action,
                np.abs(action),
                actuated_qvel,
                action * actuated_qvel,
                np.asarray(
                    [
                        height,
                        pitch,
                        height * pitch,
                        height * height,
                        pitch * pitch,
                        float(np.linalg.norm(qpos_rel)),
                        float(np.linalg.norm(qvel)),
                        float(np.linalg.norm(action)),
                        float(np.linalg.norm(actuated_qvel)),
                    ],
                    dtype=float,
                ),
            ]
        )
        return np.concatenate([base, _deterministic_rff(base, 64), np.ones(1, dtype=float)])

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(KERNEL_SUPPORT_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(KERNEL_SUPPORT_RESPONSE_PROBE_FEATURE_NAMES),
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(KERNEL_SUPPORT_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(KERNEL_SUPPORT_RESPONSE_PROBE_FEATURE_NAMES))


class _FrozenGravityResidualProbe(_FrozenForceBalanceResidualProbe):
    """Frozen model for residualized direct gravity-signature telemetry."""

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(GRAVITY_RESIDUAL_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(GRAVITY_RESIDUAL_PROBE_FEATURE_NAMES),
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(GRAVITY_RESIDUAL_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(GRAVITY_RESIDUAL_PROBE_FEATURE_NAMES))


class _FrozenGravityCompensationProbe(_FrozenGravityResidualProbe):
    """Frozen nonlinear posture model for generalized-force gravity bias."""

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        qpos_rel = np.zeros(0, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = qpos[1:] if qpos.size > 1 else qpos
        qpos_rel = _fixed_width(qpos_rel, 8)
        qvel = (
            _fixed_width(np.asarray(transition.qvel, dtype=float).reshape(-1), 9)
            if transition.qvel is not None
            else np.zeros(9, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        actuated_qvel = _actuated_qvel(qvel, action.size)
        height = float(qpos_rel[0]) if qpos_rel.size else 0.0
        pitch = float(qpos_rel[1]) if qpos_rel.size > 1 else 0.0
        return np.concatenate(
            [
                np.sin(qpos_rel),
                np.cos(qpos_rel),
                qvel,
                np.abs(qvel),
                action,
                np.abs(action),
                action * actuated_qvel,
                np.asarray(
                    [
                        height,
                        pitch,
                        height * pitch,
                        height * height,
                        pitch * pitch,
                        float(np.linalg.norm(qvel)),
                        float(np.linalg.norm(action)),
                        1.0,
                    ],
                    dtype=float,
                ),
            ]
        )

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(GRAVITY_COMPENSATION_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(GRAVITY_COMPENSATION_PROBE_FEATURE_NAMES),
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(GRAVITY_COMPENSATION_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(GRAVITY_COMPENSATION_PROBE_FEATURE_NAMES))


def _actuated_qvel(values: np.ndarray, action_dim: int = 6) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size >= action_dim:
        return arr[-action_dim:].copy()
    return _fixed_width(arr, action_dim)


def _previous_action_vector(transition: _Transition, action_dim: int = 6) -> np.ndarray:
    if transition.info and "previous_action" in transition.info:
        return _fixed_width(np.asarray(transition.info["previous_action"], dtype=float), action_dim)
    return np.zeros(action_dim, dtype=float)


def _policy_innovation_vector(transition: _Transition, action_dim: int = 6) -> np.ndarray:
    if transition.info and "policy_innovation" in transition.info:
        return _fixed_width(
            np.asarray(transition.info["policy_innovation"], dtype=float),
            action_dim,
        )
    action = _fixed_width(np.asarray(transition.action, dtype=float), action_dim)
    if transition.info and "policy_mean" in transition.info:
        policy_mean = _fixed_width(
            np.asarray(transition.info["policy_mean"], dtype=float),
            action_dim,
        )
    else:
        policy_mean = np.zeros(action_dim, dtype=float)
    policy_std = 1.0
    if transition.info and "policy_std" in transition.info:
        policy_std = float(transition.info["policy_std"])
    return (action - policy_mean) / max(abs(policy_std), 1e-8)


def _previous_policy_innovation_vector(
    transition: _Transition,
    action_dim: int = 6,
) -> np.ndarray:
    if transition.info and "previous_policy_innovation" in transition.info:
        return _fixed_width(
            np.asarray(transition.info["previous_policy_innovation"], dtype=float),
            action_dim,
        )
    return np.zeros(action_dim, dtype=float)


class _FrozenControlResponseProbe:
    """Frozen ridge model for action-to-motion and work/efficiency residuals."""

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = float(ridge)
        self.beta: np.ndarray | None = None
        self.residual_center: np.ndarray | None = None
        self.residual_scale: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.beta is not None

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        qpos_rel = np.zeros(0, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = qpos[1:] if qpos.size > 1 else qpos
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        actuated_qvel = _actuated_qvel(qvel, action.size)
        return np.concatenate(
            [
                qpos_rel,
                qvel,
                action,
                np.abs(action),
                action * actuated_qvel,
                np.ones(1, dtype=float),
            ]
        )

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        next_qvel = (
            np.asarray(transition.next_qvel, dtype=float).reshape(-1)
            if transition.next_qvel is not None
            else np.zeros(0, dtype=float)
        )
        qvel_delta = _fixed_width(next_qvel - qvel, 9)
        actuated_qvel = _actuated_qvel(qvel, action.size)
        actuated_delta = _actuated_qvel(qvel_delta, action.size)
        qpos_delta_rel = np.zeros(0, dtype=float)
        if transition.qpos is not None and transition.next_qpos is not None:
            qpos_delta = np.asarray(transition.next_qpos, dtype=float) - np.asarray(
                transition.qpos,
                dtype=float,
            )
            qpos_delta_rel = qpos_delta[1:] if qpos_delta.size > 1 else qpos_delta
        action_norm = float(np.linalg.norm(action))
        response_ratio = actuated_delta / (np.abs(action) + 5e-2)
        x_velocity = _numeric_info_value(transition.info, "x_velocity")
        reward_forward = _numeric_info_value(transition.info, "reward_forward")
        reward_ctrl = _numeric_info_value(transition.info, "reward_ctrl")
        return np.concatenate(
            [
                actuated_delta,
                action * actuated_delta,
                actuated_qvel * actuated_delta,
                response_ratio,
                np.asarray(
                    [
                        float(qpos_delta_rel[0]) if qpos_delta_rel.size > 0 else 0.0,
                        float(qpos_delta_rel[1]) if qpos_delta_rel.size > 1 else 0.0,
                        x_velocity,
                        reward_forward,
                        reward_ctrl,
                        float(transition.reward),
                        action_norm,
                        float(np.linalg.norm(actuated_qvel)),
                        float(np.linalg.norm(actuated_delta)),
                        x_velocity / (action_norm + 5e-2),
                    ],
                    dtype=float,
                ),
            ]
        )

    def fit(self, transitions: list[_Transition]) -> None:
        usable = [
            transition
            for transition in transitions
            if transition.qvel is not None and transition.next_qvel is not None
        ]
        if not usable:
            return
        x = np.vstack([self._input_vector(transition) for transition in usable])
        y = np.vstack([self._target_vector(transition) for transition in usable])
        gram = x.T @ x
        regularizer = self.ridge * np.eye(gram.shape[0], dtype=float)
        try:
            self.beta = np.linalg.solve(gram + regularizer, x.T @ y)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.lstsq(gram + regularizer, x.T @ y, rcond=None)[0]
        residuals = x @ self.beta - y
        self.residual_center = np.median(residuals, axis=0)
        q75, q25 = np.percentile(residuals, [75.0, 25.0], axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std_scale = np.std(residuals, axis=0)
        target_scale = np.std(y, axis=0)
        self.residual_scale = np.maximum.reduce(
            [
                np.asarray(iqr_scale, dtype=float),
                np.asarray(std_scale, dtype=float),
                0.1 * np.asarray(target_scale, dtype=float),
                np.full(y.shape[1], 1e-3, dtype=float),
            ]
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(CONTROL_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(CONTROL_RESPONSE_PROBE_FEATURE_NAMES))


class _FrozenGaitPhaseResponseProbe(_FrozenControlResponseProbe):
    """Frozen nonlinear gait-phase model for Walker-style motion residuals."""

    @staticmethod
    def _leg_state(transition: _Transition) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qpos = (
            np.asarray(transition.qpos, dtype=float).reshape(-1)
            if transition.qpos is not None
            else np.zeros(0, dtype=float)
        )
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        joints = _fixed_width(qpos[3:] if qpos.size > 3 else np.zeros(0, dtype=float), 6)
        joint_vel = _actuated_qvel(qvel, action.size)
        return joints, joint_vel, action

    @classmethod
    def _phase_basis(cls, transition: _Transition) -> np.ndarray:
        joints, joint_vel, action = cls._leg_state(transition)
        leg_a = joints[:3]
        leg_b = joints[3:6]
        vel_a = joint_vel[:3]
        vel_b = joint_vel[3:6]
        phase_a = float(np.arctan2(np.mean(vel_a), np.mean(leg_a) + 1e-6))
        phase_b = float(np.arctan2(np.mean(vel_b), np.mean(leg_b) + 1e-6))
        phase_diff = phase_a - phase_b
        qpos = (
            np.asarray(transition.qpos, dtype=float).reshape(-1)
            if transition.qpos is not None
            else np.zeros(0, dtype=float)
        )
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        height = float(qpos[1]) if qpos.size > 1 else 0.0
        pitch = float(qpos[2]) if qpos.size > 2 else 0.0
        root_vz = float(qvel[1]) if qvel.size > 1 else 0.0
        root_pitch_rate = float(qvel[2]) if qvel.size > 2 else 0.0
        action_norm = float(np.linalg.norm(action))
        return np.concatenate(
            [
                joints,
                joint_vel,
                action,
                np.abs(action),
                np.asarray(
                    [
                        np.sin(phase_a),
                        np.cos(phase_a),
                        np.sin(phase_b),
                        np.cos(phase_b),
                        np.sin(phase_diff),
                        np.cos(phase_diff),
                        float(np.linalg.norm(leg_a - leg_b)),
                        float(np.linalg.norm(vel_a - vel_b)),
                        float(np.linalg.norm(action[:3] - action[3:6])),
                        height,
                        pitch,
                        root_vz,
                        root_pitch_rate,
                        action_norm,
                    ],
                    dtype=float,
                ),
                np.ones(1, dtype=float),
            ]
        )

    @classmethod
    def _input_vector(cls, transition: _Transition) -> np.ndarray:
        basis = cls._phase_basis(transition)
        phase_terms = basis[24:30]
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        interactions = np.concatenate([action * term for term in phase_terms[:4]])
        return np.concatenate([basis, interactions])

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        qpos = (
            np.asarray(transition.qpos, dtype=float).reshape(-1)
            if transition.qpos is not None
            else np.zeros(0, dtype=float)
        )
        next_qpos = (
            np.asarray(transition.next_qpos, dtype=float).reshape(-1)
            if transition.next_qpos is not None
            else np.zeros(0, dtype=float)
        )
        qvel = (
            np.asarray(transition.qvel, dtype=float).reshape(-1)
            if transition.qvel is not None
            else np.zeros(0, dtype=float)
        )
        next_qvel = (
            np.asarray(transition.next_qvel, dtype=float).reshape(-1)
            if transition.next_qvel is not None
            else np.zeros(0, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        qvel_delta = _fixed_width(next_qvel - qvel, 9)
        joint_vel = _actuated_qvel(qvel, action.size)
        joint_accel = _actuated_qvel(qvel_delta, action.size)
        joints = _fixed_width(qpos[3:] if qpos.size > 3 else np.zeros(0, dtype=float), 6)
        leg_a = joints[:3]
        leg_b = joints[3:6]
        vel_a = joint_vel[:3]
        vel_b = joint_vel[3:6]
        height = float(qpos[1]) if qpos.size > 1 else 0.0
        next_height = float(next_qpos[1]) if next_qpos.size > 1 else height
        pitch = float(qpos[2]) if qpos.size > 2 else 0.0
        next_pitch = float(next_qpos[2]) if next_qpos.size > 2 else pitch
        root_vz = float(qvel[1]) if qvel.size > 1 else 0.0
        root_pitch_rate = float(qvel[2]) if qvel.size > 2 else 0.0
        action_norm = float(np.linalg.norm(action))
        denom = action_norm + 5e-2
        action_velocity_alignment = float(np.dot(action, joint_vel))
        return np.concatenate(
            [
                joint_vel,
                joint_accel,
                np.asarray(
                    [
                        next_height - height,
                        next_pitch - pitch,
                        _numeric_info_value(transition.info, "x_velocity"),
                        root_vz,
                        root_pitch_rate,
                        float(np.linalg.norm(leg_a - leg_b)),
                        float(np.linalg.norm(vel_a - vel_b)),
                        action_velocity_alignment,
                        abs(action_velocity_alignment),
                        _numeric_info_value(transition.info, "x_velocity") / denom,
                        root_vz / denom,
                        root_pitch_rate / denom,
                    ],
                    dtype=float,
                ),
            ]
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(GAIT_PHASE_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(GAIT_PHASE_RESPONSE_PROBE_FEATURE_NAMES))


class _FrozenGaitLoadResponseProbe(_FrozenGaitPhaseResponseProbe):
    """Frozen gait-phase model for support-load and vertical-response residuals."""

    @classmethod
    def _input_vector(cls, transition: _Transition) -> np.ndarray:
        basis = cls._phase_basis(transition)
        phase_terms = basis[24:30]
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        qvel = (
            _fixed_width(np.asarray(transition.qvel, dtype=float).reshape(-1), 9)
            if transition.qvel is not None
            else np.zeros(9, dtype=float)
        )
        actuated_qvel = _actuated_qvel(qvel, action.size)
        interactions = np.concatenate(
            [
                action * term
                for term in phase_terms[:4]
            ]
            + [
                actuated_qvel * phase_terms[4],
                actuated_qvel * phase_terms[5],
                action * actuated_qvel,
            ]
        )
        return np.concatenate([basis, interactions])

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(GAIT_LOAD_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(GAIT_LOAD_RESPONSE_PROBE_FEATURE_NAMES),
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(GAIT_LOAD_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(GAIT_LOAD_RESPONSE_PROBE_FEATURE_NAMES))


class _FrozenWalkerGravityResponseProbe(_FrozenGravityCompensationProbe):
    """Frozen Walker posture/gait model for vertical gravity-response residuals."""

    @classmethod
    def _input_vector(cls, transition: _Transition) -> np.ndarray:
        phase_basis = _FrozenGaitPhaseResponseProbe._phase_basis(transition)
        qpos_rel = np.zeros(8, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = _fixed_width(qpos[1:] if qpos.size > 1 else qpos, 8)
        qvel = (
            _fixed_width(np.asarray(transition.qvel, dtype=float).reshape(-1), 9)
            if transition.qvel is not None
            else np.zeros(9, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        actuated_qvel = _actuated_qvel(qvel, action.size)
        height = float(qpos_rel[0]) if qpos_rel.size else 0.0
        pitch = float(qpos_rel[1]) if qpos_rel.size > 1 else 0.0
        phase_terms = phase_basis[24:30]
        base = np.concatenate(
            [
                qpos_rel,
                np.sin(qpos_rel),
                np.cos(qpos_rel),
                qvel,
                np.abs(qvel),
                action,
                np.abs(action),
                actuated_qvel,
                action * actuated_qvel,
                phase_basis,
                np.concatenate([action * term for term in phase_terms[:4]]),
                np.asarray(
                    [
                        height,
                        pitch,
                        height * pitch,
                        height * height,
                        pitch * pitch,
                        float(np.linalg.norm(qpos_rel)),
                        float(np.linalg.norm(qvel)),
                        float(np.linalg.norm(action)),
                        float(np.linalg.norm(actuated_qvel)),
                    ],
                    dtype=float,
                ),
            ]
        )
        return np.concatenate([base, _deterministic_rff(base, 48), np.ones(1, dtype=float)])

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        if transition.physics is None:
            return np.zeros(len(WALKER_GRAVITY_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        return _fixed_width(
            np.asarray(transition.physics, dtype=float),
            len(WALKER_GRAVITY_RESPONSE_PROBE_FEATURE_NAMES),
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(WALKER_GRAVITY_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(WALKER_GRAVITY_RESPONSE_PROBE_FEATURE_NAMES))


class _FrozenWalkerExcitationResponseProbe(_FrozenControlResponseProbe):
    """Frozen model for response to ordinary-policy action innovations."""

    @classmethod
    def _input_vector(cls, transition: _Transition) -> np.ndarray:
        phase_basis = _FrozenGaitPhaseResponseProbe._phase_basis(transition)
        qpos_rel = np.zeros(8, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = _fixed_width(qpos[1:] if qpos.size > 1 else qpos, 8)
        qvel = (
            _fixed_width(np.asarray(transition.qvel, dtype=float).reshape(-1), 9)
            if transition.qvel is not None
            else np.zeros(9, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        previous_action = _previous_action_vector(transition, action.size)
        action_delta = action - previous_action
        actuated_qvel = _actuated_qvel(qvel, action.size)
        phase_terms = phase_basis[24:30]
        base = np.concatenate(
            [
                qpos_rel,
                np.sin(qpos_rel),
                np.cos(qpos_rel),
                qvel,
                np.abs(qvel),
                action,
                previous_action,
                action_delta,
                np.abs(action_delta),
                actuated_qvel,
                action * actuated_qvel,
                action_delta * actuated_qvel,
                phase_basis,
                np.concatenate([action_delta * term for term in phase_terms[:4]]),
                np.asarray(
                    [
                        float(np.linalg.norm(action)),
                        float(np.linalg.norm(previous_action)),
                        float(np.linalg.norm(action_delta)),
                        float(np.linalg.norm(actuated_qvel)),
                        float(np.dot(action_delta, actuated_qvel)),
                    ],
                    dtype=float,
                ),
            ]
        )
        return np.concatenate([base, _deterministic_rff(base, 48), np.ones(1, dtype=float)])

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        return _walker_excitation_response_probe_target_vector(transition)

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(WALKER_EXCITATION_RESPONSE_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(WALKER_EXCITATION_RESPONSE_PROBE_FEATURE_NAMES))


class _FrozenWalkerInnovationGainProbe(_FrozenControlResponseProbe):
    """Frozen model for gait-stratified policy-innovation response gains."""

    @classmethod
    def _input_vector(cls, transition: _Transition) -> np.ndarray:
        phase_basis = _FrozenGaitPhaseResponseProbe._phase_basis(transition)
        qpos_rel = np.zeros(8, dtype=float)
        if transition.qpos is not None:
            qpos = np.asarray(transition.qpos, dtype=float).reshape(-1)
            qpos_rel = _fixed_width(qpos[1:] if qpos.size > 1 else qpos, 8)
        qvel = (
            _fixed_width(np.asarray(transition.qvel, dtype=float).reshape(-1), 9)
            if transition.qvel is not None
            else np.zeros(9, dtype=float)
        )
        action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
        previous_action = _previous_action_vector(transition, action.size)
        action_delta = action - previous_action
        innovation = _policy_innovation_vector(transition, action.size)
        previous_innovation = _previous_policy_innovation_vector(
            transition,
            action.size,
        )
        actuated_qvel = _actuated_qvel(qvel, action.size)
        phase_terms = phase_basis[24:30]
        base = np.concatenate(
            [
                qpos_rel,
                np.sin(qpos_rel),
                np.cos(qpos_rel),
                qvel,
                np.abs(qvel),
                action,
                previous_action,
                action_delta,
                innovation,
                previous_innovation,
                actuated_qvel,
                action * actuated_qvel,
                innovation * actuated_qvel,
                phase_basis,
                np.concatenate([innovation * term for term in phase_terms[:4]]),
                np.asarray(
                    [
                        float(np.linalg.norm(action)),
                        float(np.linalg.norm(action_delta)),
                        float(np.linalg.norm(innovation)),
                        float(np.linalg.norm(previous_innovation)),
                        float(np.linalg.norm(actuated_qvel)),
                        float(np.dot(innovation, actuated_qvel)),
                    ],
                    dtype=float,
                ),
            ]
        )
        return np.concatenate([base, _deterministic_rff(base, 48), np.ones(1, dtype=float)])

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        return _walker_innovation_gain_probe_target_vector(transition)

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(WALKER_INNOVATION_GAIN_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(WALKER_INNOVATION_GAIN_PROBE_FEATURE_NAMES))


class _FrozenSensorConsistencyProbe:
    """Frozen redundant-state model for observation readout residuals."""

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = float(ridge)
        self.beta: np.ndarray | None = None
        self.residual_center: np.ndarray | None = None
        self.residual_scale: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.beta is not None

    @staticmethod
    def _input_vector(transition: _Transition) -> np.ndarray:
        qpos_rel = np.zeros(0, dtype=float)
        if transition.next_qpos is not None:
            qpos = np.asarray(transition.next_qpos, dtype=float).reshape(-1)
            qpos_rel = qpos[1:] if qpos.size > 1 else qpos
        qvel = (
            np.asarray(transition.next_qvel, dtype=float).reshape(-1)
            if transition.next_qvel is not None
            else np.zeros(0, dtype=float)
        )
        return np.concatenate([qpos_rel, qvel, np.ones(1, dtype=float)])

    @staticmethod
    def _target_vector(transition: _Transition) -> np.ndarray:
        return _fixed_width(np.asarray(transition.next_obs, dtype=float), 17)

    def fit(self, transitions: list[_Transition]) -> None:
        usable = [
            transition
            for transition in transitions
            if transition.next_qpos is not None and transition.next_qvel is not None
        ]
        if not usable:
            return
        x = np.vstack([self._input_vector(transition) for transition in usable])
        y = np.vstack([self._target_vector(transition) for transition in usable])
        gram = x.T @ x
        regularizer = self.ridge * np.eye(gram.shape[0], dtype=float)
        try:
            self.beta = np.linalg.solve(gram + regularizer, x.T @ y)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.lstsq(gram + regularizer, x.T @ y, rcond=None)[0]
        residuals = x @ self.beta - y
        self.residual_center = np.median(residuals, axis=0)
        q75, q25 = np.percentile(residuals, [75.0, 25.0], axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std_scale = np.std(residuals, axis=0)
        target_scale = np.std(y, axis=0)
        self.residual_scale = np.maximum.reduce(
            [
                np.asarray(iqr_scale, dtype=float),
                np.asarray(std_scale, dtype=float),
                0.1 * np.asarray(target_scale, dtype=float),
                np.full(y.shape[1], 1e-3, dtype=float),
            ]
        )

    def feature_vector(self, transition: _Transition) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(SENSOR_CONSISTENCY_PROBE_FEATURE_NAMES), dtype=float)
        assert self.beta is not None
        assert self.residual_center is not None
        assert self.residual_scale is not None
        x = self._input_vector(transition)
        residual = (x @ self.beta) - self._target_vector(transition)
        residual = (residual - self.residual_center) / self.residual_scale
        residual = np.sign(residual) * np.log1p(np.abs(residual))
        return _fixed_width(residual, len(SENSOR_CONSISTENCY_PROBE_FEATURE_NAMES))


def _require_gymnasium():
    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo benchmarks require optional dependencies. Install them with "
            "`python -m pip install 'gymnasium[mujoco]'` or `python -m pip install -e '.[mujoco]'`."
        ) from exc
    return gym


def _current_observation(env) -> np.ndarray | None:
    getter = getattr(env.unwrapped, "_get_obs", None)
    if callable(getter):
        return np.asarray(getter(), dtype=float)
    return None


def _get_task_target(env) -> np.ndarray | None:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "goal"):
        return np.asarray(unwrapped.goal, dtype=float).reshape(-1).copy()
    if hasattr(unwrapped, "goal_pos"):
        return np.asarray(unwrapped.goal_pos, dtype=float).reshape(-1).copy()
    return None


def _set_task_target(env, target: np.ndarray) -> np.ndarray | None:
    unwrapped = env.unwrapped
    values = np.asarray(target, dtype=float).reshape(-1).copy()
    if values.size < 2:
        raise AttributeError("MuJoCo task target has fewer than two coordinates")
    qpos = np.asarray(unwrapped.data.qpos, dtype=float).copy()
    qvel = np.asarray(unwrapped.data.qvel, dtype=float).copy()
    if hasattr(unwrapped, "goal"):
        unwrapped.goal = values.copy()
    elif hasattr(unwrapped, "goal_pos"):
        unwrapped.goal_pos = values.copy()
    else:
        raise AttributeError("MuJoCo environment has no goal/goal_pos field for task target")
    if qpos.size >= 2:
        qpos[-2:] = values[:2]
        unwrapped.set_state(qpos, qvel)
    try:
        import mujoco

        mujoco.mj_forward(unwrapped.model, unwrapped.data)
    except Exception:
        pass
    return _current_observation(env)


def _apply_target_shift(env, target_shift: float) -> None:
    target = _get_task_target(env)
    if target is None:
        raise AttributeError("MuJoCo environment has no goal/goal_pos field for target shift")
    offset = np.asarray([target_shift, -target_shift], dtype=float)
    shifted = target.copy()
    shifted[:2] = shifted[:2] + offset
    _set_task_target(env, shifted)


def _apply_shift(
    env,
    shift: MujocoShift,
    gravity_scale: float,
    friction_scale: float,
    target_shift: float,
) -> None:
    if shift == "no_shift":
        return
    unwrapped = env.unwrapped
    if shift == "gravity":
        unwrapped.model.opt.gravity[:] = np.asarray(unwrapped.model.opt.gravity) * np.asarray(
            [1.0, 1.0, gravity_scale]
        )
    elif shift == "friction":
        if hasattr(unwrapped.model, "geom_friction"):
            unwrapped.model.geom_friction[:, 0] *= friction_scale
        else:
            raise AttributeError("MuJoCo model has no geom_friction field for friction shift")
    elif shift == "target":
        _apply_target_shift(env, target_shift)
    elif shift == "sensor_bias":
        return
    elif shift == "actuator_loss":
        return
    else:
        raise ValueError(f"unknown MuJoCo shift {shift!r}")
    try:
        import mujoco

        mujoco.mj_forward(unwrapped.model, unwrapped.data)
    except Exception:
        pass


def _scripted_policy_action(
    mode: MujocoPolicyMode,
    *,
    t: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    env_id: str,
) -> np.ndarray:
    if mode == "zero":
        return np.clip(np.zeros_like(action_low, dtype=float), action_low, action_high)
    if mode not in {"sinusoidal", "chirp", "pulse"}:
        raise ValueError(f"scripted action requested for unsupported policy mode {mode!r}")

    action_dim = int(action_low.size)
    center = 0.5 * (action_low + action_high)
    radius = 0.5 * (action_high - action_low)
    def match_width(values: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == action_dim:
            return arr
        return np.resize(arr, action_dim)

    if mode == "sinusoidal":
        offsets = np.linspace(0.0, np.pi, action_dim, endpoint=False)
        if env_id.startswith("Walker2d"):
            frequency = 0.18
            amplitude = 0.45
            offsets = match_width([0.0, 1.8, 3.1, 0.9, 2.6, 4.0])
        elif env_id.startswith("HalfCheetah"):
            frequency = 0.22
            amplitude = 0.55
            offsets = match_width([0.0, 1.2, 2.4, 3.3, 4.2, 5.1])
        else:
            frequency = 0.16
            amplitude = 0.35
        unit = np.sin(frequency * float(t) + offsets)
    elif mode == "chirp":
        offsets = np.linspace(0.0, 2.0 * np.pi, action_dim, endpoint=False)
        if env_id.startswith("Walker2d"):
            base_frequency = 0.035
            chirp_rate = 4.5e-5
            amplitude = 0.33
            offsets = match_width([0.0, 2.2, 4.1, 1.0, 3.0, 5.2])
        elif env_id.startswith("HalfCheetah"):
            base_frequency = 0.030
            chirp_rate = 5.0e-5
            amplitude = 0.38
            offsets = match_width([0.0, 1.0, 2.1, 3.4, 4.6, 5.7])
        else:
            base_frequency = 0.030
            chirp_rate = 4.0e-5
            amplitude = 0.30
        phase = base_frequency * float(t) + chirp_rate * float(t) ** 2
        slow_phase = 0.013 * float(t) + 2.0 * offsets
        unit = 0.78 * np.sin(phase + offsets) + 0.22 * np.sin(slow_phase)
    else:
        offsets = np.linspace(0.0, 2.0 * np.pi, action_dim, endpoint=False)
        if env_id.startswith("Walker2d"):
            period = 70
            width = 14
            amplitude = 0.28
            pattern = match_width([1.0, -1.0, 1.0, -1.0, 0.8, -0.8])
        elif env_id.startswith("HalfCheetah"):
            period = 72
            width = 12
            amplitude = 0.34
            pattern = match_width([1.0, -1.0, -0.8, 0.8, 1.0, -1.0])
        else:
            period = 80
            width = 12
            amplitude = 0.25
            pattern = np.where(np.arange(action_dim) % 2 == 0, 1.0, -1.0)
        cycle = int(t) % period
        sign = 1.0 if (int(t) // period) % 2 == 0 else -1.0
        if cycle < width:
            envelope = np.sin(np.pi * float(cycle + 1) / float(width + 1))
        elif cycle < 2 * width:
            envelope = -0.4 * np.sin(np.pi * float(cycle - width + 1) / float(width + 1))
        else:
            envelope = 0.0
        dither = 0.08 * np.sin(0.09 * float(t) + offsets)
        unit = sign * envelope * pattern + dither
    raw = center + amplitude * radius * unit
    return np.clip(raw, action_low, action_high)


def _linear_sinusoidal_dither_action(
    agent: _LinearActorValue,
    obs: np.ndarray,
    *,
    t: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    env_id: str,
) -> np.ndarray:
    base_action = agent.act(obs)
    center = 0.5 * (action_low + action_high)
    scripted = _scripted_policy_action(
        "sinusoidal",
        t=t,
        action_low=action_low,
        action_high=action_high,
        env_id=env_id,
    )
    dither = 0.60 * (scripted - center)
    return np.clip(base_action + dither, action_low, action_high)


def _linear_probe_burst_action(
    agent: _LinearActorValue,
    obs: np.ndarray,
    *,
    t: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    env_id: str,
) -> np.ndarray:
    """Linear actor with sparse open-loop probe bursts for active monitoring."""

    base_action = agent.act(obs)
    if env_id.startswith("Walker2d"):
        period = 96
        width = 18
        amplitude = 0.78
    elif env_id.startswith("HalfCheetah"):
        period = 104
        width = 16
        amplitude = 0.72
    else:
        period = 96
        width = 14
        amplitude = 0.55
    cycle = int(t) % period
    if cycle >= width:
        return base_action
    center = 0.5 * (action_low + action_high)
    scripted = _scripted_policy_action(
        "sinusoidal",
        t=t,
        action_low=action_low,
        action_high=action_high,
        env_id=env_id,
    )
    envelope = np.sin(np.pi * float(cycle + 1) / float(width + 1))
    burst = amplitude * envelope * (scripted - center)
    return np.clip(base_action + burst, action_low, action_high)


def _mujoco_array_norm(env, name: str, *, drop_first: bool = False) -> float:
    try:
        values = getattr(env.unwrapped.data, name)
    except Exception:
        return 0.0
    arr = np.asarray(values, dtype=float).reshape(-1)
    if drop_first and arr.size > 1:
        arr = arr[1:]
    return float(np.linalg.norm(arr))


def _mujoco_data_array(env, name: str) -> np.ndarray:
    try:
        values = getattr(env.unwrapped.data, name)
    except Exception:
        return np.zeros(0, dtype=float)
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _fixed_width(values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = np.zeros(width, dtype=float)
    n = min(width, arr.size)
    if n:
        out[:n] = np.nan_to_num(arr[:n], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _deterministic_rff(values: np.ndarray, components: int = 64) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or components <= 0:
        return np.zeros(max(int(components), 0), dtype=float)
    rows = np.arange(1, arr.size + 1, dtype=float)[:, None]
    cols = np.arange(1, int(components) + 1, dtype=float)[None, :]
    weights = np.sin(12.9898 * rows + 78.233 * cols) / np.sqrt(float(arr.size))
    phases = 2.0 * np.pi * np.mod(np.sin(37.719 * cols).reshape(-1) * 43758.5453, 1.0)
    projection = arr @ weights + phases
    return np.sqrt(2.0 / float(components)) * np.cos(projection)


def _signed_log_ratio(numerator: float, denominator: float, eps: float = 5e-2) -> float:
    ratio = abs(float(numerator)) / (abs(float(denominator)) + float(eps))
    return float(np.sign(numerator) * np.log1p(ratio))


def _sensor_bias_vector(obs_dim: int, bias_scale: float, seed: int) -> np.ndarray:
    """Deterministic post-change sensor offset with bounded coordinate magnitude."""

    dim = int(obs_dim)
    if dim <= 0:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(int(seed) + 104729)
    pattern = rng.normal(size=dim)
    max_abs = float(np.max(np.abs(pattern)))
    if max_abs <= 1e-12:
        pattern = np.ones(dim, dtype=float)
        max_abs = 1.0
    return float(bias_scale) * pattern / max_abs


def _apply_sensor_bias(obs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=float)
    flat = arr.reshape(-1)
    offset = _fixed_width(np.asarray(bias, dtype=float), flat.size)
    return (flat + offset).reshape(arr.shape)


def _apply_actuator_loss(action: np.ndarray, actuator_loss_scale: float) -> np.ndarray:
    return np.asarray(action, dtype=float) * float(actuator_loss_scale)


def _physics_probe_target_vector(env) -> np.ndarray:
    cfrc = _mujoco_data_array(env, "cfrc_ext")
    cfrc_norms = (
        np.linalg.norm(cfrc.reshape(cfrc.shape[0], -1), axis=1)
        if cfrc.ndim >= 2 and cfrc.size
        else np.zeros(0, dtype=float)
    )
    qfrc_constraint = _mujoco_data_array(env, "qfrc_constraint").reshape(-1)
    qfrc_passive = _mujoco_data_array(env, "qfrc_passive").reshape(-1)
    qacc = _mujoco_data_array(env, "qacc").reshape(-1)
    return np.concatenate(
        [
            _fixed_width(cfrc_norms, 8),
            _fixed_width(qfrc_constraint, 9),
            _fixed_width(qfrc_passive, 9),
            _fixed_width(qacc, 9),
        ]
    )


def _force_balance_probe_target_vector(env) -> np.ndarray:
    cfrc = _mujoco_data_array(env, "cfrc_ext").reshape(-1)
    qfrc_constraint = _mujoco_data_array(env, "qfrc_constraint").reshape(-1)
    qfrc_passive = _mujoco_data_array(env, "qfrc_passive").reshape(-1)
    qfrc_bias = _mujoco_data_array(env, "qfrc_bias").reshape(-1)
    qfrc_actuator = _mujoco_data_array(env, "qfrc_actuator").reshape(-1)
    qacc = _mujoco_data_array(env, "qacc").reshape(-1)
    return np.concatenate(
        [
            _fixed_width(cfrc, 48),
            _fixed_width(qfrc_constraint, 9),
            _fixed_width(qfrc_passive, 9),
            _fixed_width(qfrc_bias, 9),
            _fixed_width(qfrc_actuator, 9),
            _fixed_width(qacc, 9),
        ]
    )


def _contact_distances(env) -> np.ndarray:
    try:
        data = env.unwrapped.data
        ncon = int(getattr(data, "ncon", 0))
        contacts = getattr(data, "contact")
    except Exception:
        return np.zeros(0, dtype=float)
    distances: list[float] = []
    for index in range(max(ncon, 0)):
        try:
            distances.append(float(contacts[index].dist))
        except Exception:
            continue
    return np.nan_to_num(np.asarray(distances, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _contact_slip_probe_feature_vector(env) -> np.ndarray:
    xpos = _mujoco_data_array(env, "xpos")
    body_z = xpos[:, 2] if xpos.ndim == 2 and xpos.shape[1] >= 3 else np.zeros(0, dtype=float)
    cvel = _mujoco_data_array(env, "cvel")
    if cvel.ndim == 2 and cvel.shape[1] >= 6:
        linear_velocity = cvel[:, 3:6]
    elif cvel.ndim == 2 and cvel.shape[1] >= 3:
        linear_velocity = cvel[:, -3:]
    else:
        linear_velocity = np.zeros((0, 3), dtype=float)
    xy_speed = (
        np.linalg.norm(linear_velocity[:, :2], axis=1)
        if linear_velocity.size
        else np.zeros(0, dtype=float)
    )
    vertical_speed = (
        linear_velocity[:, 2]
        if linear_velocity.ndim == 2 and linear_velocity.shape[1] >= 3
        else np.zeros(0, dtype=float)
    )
    cfrc = _mujoco_data_array(env, "cfrc_ext")
    load_norm = (
        np.linalg.norm(cfrc.reshape(cfrc.shape[0], -1), axis=1)
        if cfrc.ndim >= 2 and cfrc.size
        else np.zeros(0, dtype=float)
    )
    width = 8
    body_z_fixed = _fixed_width(body_z, width)
    xy_speed_fixed = _fixed_width(xy_speed, width)
    vertical_speed_fixed = _fixed_width(vertical_speed, width)
    load_norm_fixed = _fixed_width(load_norm, width)
    loaded_xy_speed = xy_speed_fixed * np.log1p(load_norm_fixed)
    low_height_weight = 1.0 / (1.0 + np.exp(12.0 * (body_z_fixed - 0.35)))
    low_height_xy_speed = xy_speed_fixed * low_height_weight
    distances = _contact_distances(env)
    if distances.size:
        dist_min = float(np.min(distances))
        dist_mean = float(np.mean(distances))
        dist_max = float(np.max(distances))
    else:
        dist_min = dist_mean = dist_max = 0.0
    try:
        contact_count = float(int(getattr(env.unwrapped.data, "ncon", 0)))
    except Exception:
        contact_count = 0.0
    return np.concatenate(
        [
            np.asarray(
                [
                    contact_count,
                    dist_min,
                    dist_mean,
                    dist_max,
                    float(np.sum(load_norm_fixed)),
                    float(np.sum(loaded_xy_speed)),
                ],
                dtype=float,
            ),
            body_z_fixed,
            xy_speed_fixed,
            vertical_speed_fixed,
            load_norm_fixed,
            loaded_xy_speed,
            low_height_xy_speed,
        ]
    )


def _support_response_probe_target_vector(transition: _Transition, env) -> np.ndarray:
    action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
    action_norm = float(np.linalg.norm(action))
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    next_qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    qvel_delta = _fixed_width(next_qvel - qvel, 9)
    actuated_delta = _actuated_qvel(qvel_delta, action.size)
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    next_qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    height = float(qpos[1]) if qpos.size > 1 else 0.0
    next_height = float(next_qpos[1]) if next_qpos.size > 1 else height
    pitch = float(qpos[2]) if qpos.size > 2 else 0.0
    next_pitch = float(next_qpos[2]) if next_qpos.size > 2 else pitch

    cfrc = _mujoco_data_array(env, "cfrc_ext")
    load_norm = (
        np.linalg.norm(cfrc.reshape(cfrc.shape[0], -1), axis=1)
        if cfrc.ndim >= 2 and cfrc.size
        else np.zeros(0, dtype=float)
    )
    load_fixed = _fixed_width(load_norm, 8)
    load_total = float(np.sum(load_fixed))
    load_share = load_fixed / (load_total + 5e-2)
    load_entropy = 0.0
    if load_total > 0.0:
        probabilities = load_fixed / max(load_total, 1e-12)
        positive = probabilities[probabilities > 0.0]
        if positive.size:
            load_entropy = float(-np.sum(positive * np.log(positive)) / np.log(load_fixed.size))
    load_concentration = float(np.max(load_share)) if load_share.size else 0.0

    cvel = _mujoco_data_array(env, "cvel")
    if cvel.ndim == 2 and cvel.shape[1] >= 6:
        linear_velocity = cvel[:, 3:6]
    elif cvel.ndim == 2 and cvel.shape[1] >= 3:
        linear_velocity = cvel[:, -3:]
    else:
        linear_velocity = np.zeros((0, 3), dtype=float)
    xy_speed = (
        np.linalg.norm(linear_velocity[:, :2], axis=1)
        if linear_velocity.size
        else np.zeros(0, dtype=float)
    )
    xy_speed_fixed = _fixed_width(xy_speed, 8)
    xpos = _mujoco_data_array(env, "xpos")
    body_z = xpos[:, 2] if xpos.ndim == 2 and xpos.shape[1] >= 3 else np.zeros(0, dtype=float)
    body_z_fixed = _fixed_width(body_z, 8)
    low_height_weight = 1.0 / (1.0 + np.exp(12.0 * (body_z_fixed - 0.35)))
    loaded_speed = xy_speed_fixed * np.log1p(load_fixed)
    loaded_speed_total = float(np.sum(loaded_speed))
    loaded_speed_share = loaded_speed / (loaded_speed_total + 5e-2)
    low_height_speed_total = float(np.sum(xy_speed_fixed * low_height_weight))

    qacc = _fixed_width(_mujoco_data_array(env, "qacc").reshape(-1), 9)
    qfrc_bias = _fixed_width(_mujoco_data_array(env, "qfrc_bias").reshape(-1), 9)
    qfrc_passive = _fixed_width(_mujoco_data_array(env, "qfrc_passive").reshape(-1), 9)
    qfrc_constraint = _fixed_width(_mujoco_data_array(env, "qfrc_constraint").reshape(-1), 9)
    qacc_norm = float(np.linalg.norm(qacc))
    bias_norm = float(np.linalg.norm(qfrc_bias))
    passive_norm = float(np.linalg.norm(qfrc_passive))
    constraint_norm = float(np.linalg.norm(qfrc_constraint))
    bias_qacc_alignment = float(
        np.dot(qfrc_bias, qacc) / ((bias_norm * qacc_norm) + 5e-2)
    )

    distances = _contact_distances(env)
    penetrations = np.maximum(-distances, 0.0) if distances.size else np.zeros(0, dtype=float)
    try:
        contact_count = float(int(getattr(env.unwrapped.data, "ncon", 0)))
    except Exception:
        contact_count = 0.0
    penetration_mean = float(np.mean(penetrations)) if penetrations.size else 0.0
    penetration_max = float(np.max(penetrations)) if penetrations.size else 0.0

    action_denom = action_norm + 5e-2
    load_denom = load_total + 5e-2
    return np.concatenate(
        [
            np.asarray(
                [
                    load_total,
                    load_total / action_denom,
                    load_total / (qacc_norm + 5e-2),
                    loaded_speed_total,
                    loaded_speed_total / load_denom,
                    low_height_speed_total,
                    contact_count,
                    penetration_mean,
                    penetration_max,
                    load_entropy,
                    load_concentration,
                    qacc_norm,
                    float(np.linalg.norm(actuated_delta)),
                    float(qvel_delta[2]) if qvel_delta.size > 2 else 0.0,
                    next_height - height,
                    next_pitch - pitch,
                    bias_norm / load_denom,
                    passive_norm / load_denom,
                    constraint_norm / load_denom,
                    bias_qacc_alignment,
                ],
                dtype=float,
            ),
            load_share,
            loaded_speed_share,
        ]
    )


def _kinematic_feature_vector(transition: _Transition, env) -> np.ndarray:
    obs = np.asarray(transition.obs, dtype=float)
    next_obs = np.asarray(transition.next_obs, dtype=float)
    action = np.asarray(transition.action, dtype=float)
    return np.asarray(
        [
            float(transition.reward),
            float(np.linalg.norm(obs)),
            float(np.linalg.norm(next_obs - obs)),
            float(np.linalg.norm(action)),
            _mujoco_array_norm(env, "qpos", drop_first=True),
            _mujoco_array_norm(env, "qvel"),
        ],
        dtype=float,
    )


def _numeric_info_value(info: dict[str, Any] | None, key: str) -> float:
    if not info or key not in info:
        return 0.0
    try:
        value = float(info[key])
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


def _telemetry_feature_vector(
    transition: _Transition,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [_numeric_info_value(transition.info, key) for key in feature_names],
        dtype=float,
    )


def _simulator_state(env) -> tuple[np.ndarray, np.ndarray]:
    data = env.unwrapped.data
    return (
        np.asarray(data.qpos, dtype=float).copy(),
        np.asarray(data.qvel, dtype=float).copy(),
    )


def _reset_probe_feature_vector(
    transitions: list[_Transition],
    probe_env,
) -> np.ndarray:
    if not transitions:
        return np.zeros(len(RESET_PROBE_FEATURE_NAMES), dtype=float)

    reward_deltas: list[float] = []
    obs_delta_norms: list[float] = []
    qpos_delta_norms: list[float] = []
    qvel_delta_norms: list[float] = []
    core = probe_env.unwrapped
    for transition in transitions:
        if (
            transition.qpos is None
            or transition.qvel is None
            or transition.next_qpos is None
            or transition.next_qvel is None
        ):
            continue
        core.set_state(transition.qpos, transition.qvel)
        next_obs, reward, _, _, _ = core.step(transition.action)
        qpos, qvel = _simulator_state(probe_env)
        reward_deltas.append(float(reward) - transition.reward)
        obs_delta_norms.append(float(np.linalg.norm(np.asarray(next_obs) - transition.next_obs)))
        qpos_delta_norms.append(float(np.linalg.norm(qpos - transition.next_qpos)))
        qvel_delta_norms.append(float(np.linalg.norm(qvel - transition.next_qvel)))

    if not reward_deltas:
        return np.zeros(len(RESET_PROBE_FEATURE_NAMES), dtype=float)
    return np.asarray(
        [
            float(np.mean(reward_deltas)),
            float(np.mean(obs_delta_norms)),
            float(np.mean(qpos_delta_norms)),
            float(np.mean(qvel_delta_norms)),
        ],
        dtype=float,
    )


def _motion_probe_feature_vector(transition: _Transition) -> np.ndarray:
    obs_delta = np.asarray(transition.next_obs, dtype=float) - np.asarray(
        transition.obs,
        dtype=float,
    )
    qpos_delta_rel = np.zeros(0, dtype=float)
    if transition.qpos is not None and transition.next_qpos is not None:
        qpos_delta_rel = np.asarray(transition.next_qpos, dtype=float) - np.asarray(
            transition.qpos,
            dtype=float,
        )
        if qpos_delta_rel.size > 1:
            qpos_delta_rel = qpos_delta_rel[1:]
    qvel_delta = np.zeros(0, dtype=float)
    if transition.qvel is not None and transition.next_qvel is not None:
        qvel_delta = np.asarray(transition.next_qvel, dtype=float) - np.asarray(
            transition.qvel,
            dtype=float,
        )
    return np.asarray(
        [
            float(np.linalg.norm(obs_delta)),
            float(np.linalg.norm(qpos_delta_rel)),
            float(np.linalg.norm(qvel_delta)),
            float(np.max(np.abs(qvel_delta))) if qvel_delta.size else 0.0,
        ],
        dtype=float,
    )


def _locomotion_observation_probe_feature_vector(transition: _Transition) -> np.ndarray:
    obs = np.asarray(transition.next_obs, dtype=float)
    obs_delta = np.asarray(transition.next_obs, dtype=float) - np.asarray(
        transition.obs,
        dtype=float,
    )
    qpos_rel = np.zeros(0, dtype=float)
    if transition.next_qpos is not None:
        qpos = np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        qpos_rel = qpos[1:] if qpos.size > 1 else qpos
    qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    action = np.asarray(transition.action, dtype=float)
    return np.concatenate(
        [
            _fixed_width(obs, 17),
            _fixed_width(obs_delta, 17),
            _fixed_width(qpos_rel, 8),
            _fixed_width(qvel, 9),
            _fixed_width(action, 6),
            np.asarray(
                [
                    float(transition.reward),
                    _numeric_info_value(transition.info, "x_velocity"),
                    _numeric_info_value(transition.info, "reward_forward"),
                    _numeric_info_value(transition.info, "reward_ctrl"),
                ],
                dtype=float,
            ),
        ]
    )


def _locomotion_dynamics_probe_feature_vector(transition: _Transition, env) -> np.ndarray:
    physics = _physics_probe_target_vector(env)
    cfrc_norms = physics[:8]
    qfrc_constraint = physics[8:17]
    qfrc_passive = physics[17:26]
    qacc = physics[26:35]
    cfrc = _mujoco_data_array(env, "cfrc_ext")
    qacc = _mujoco_data_array(env, "qacc").reshape(-1)
    qvel = _mujoco_data_array(env, "qvel").reshape(-1)
    return np.concatenate(
        [
            _fixed_width(cfrc_norms, 8),
            _fixed_width(qfrc_constraint, 9),
            _fixed_width(qfrc_passive, 9),
            _fixed_width(qacc, 9),
            _fixed_width(qvel, 9),
            np.asarray(
                [
                    float(np.linalg.norm(cfrc)),
                    float(np.linalg.norm(qfrc_constraint)),
                    float(np.linalg.norm(qfrc_passive)),
                    float(np.linalg.norm(qacc)),
                    float(transition.reward),
                    _numeric_info_value(transition.info, "x_velocity"),
                    _numeric_info_value(transition.info, "reward_forward"),
                    _numeric_info_value(transition.info, "reward_ctrl"),
                ],
                dtype=float,
            ),
        ]
    )


def _locomotion_stability_probe_feature_vector(
    transition: _Transition,
    episode_step: int,
) -> np.ndarray:
    qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    obs_delta = np.asarray(transition.next_obs, dtype=float) - np.asarray(
        transition.obs,
        dtype=float,
    )
    return np.asarray(
        [
            float(transition.done),
            min(float(episode_step), 1000.0) / 1000.0,
            float(transition.reward),
            _numeric_info_value(transition.info, "x_velocity"),
            _numeric_info_value(transition.info, "reward_forward"),
            _numeric_info_value(transition.info, "reward_ctrl"),
            float(qpos[1]) if qpos.size > 1 else 0.0,
            float(qpos[2]) if qpos.size > 2 else 0.0,
            float(np.linalg.norm(qvel)),
            float(np.linalg.norm(obs_delta)),
        ],
        dtype=float,
    )


def _gravity_signature_probe_feature_vector(transition: _Transition, env) -> np.ndarray:
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    next_qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
    qfrc_bias = _fixed_width(_mujoco_data_array(env, "qfrc_bias").reshape(-1), 9)
    qfrc_passive = _fixed_width(_mujoco_data_array(env, "qfrc_passive").reshape(-1), 9)
    qfrc_constraint = _fixed_width(_mujoco_data_array(env, "qfrc_constraint").reshape(-1), 9)
    qfrc_actuator = _fixed_width(_mujoco_data_array(env, "qfrc_actuator").reshape(-1), 9)
    qacc = _fixed_width(_mujoco_data_array(env, "qacc").reshape(-1), 9)

    cfrc = _mujoco_data_array(env, "cfrc_ext")
    load_norm = (
        np.linalg.norm(cfrc.reshape(cfrc.shape[0], -1), axis=1)
        if cfrc.ndim >= 2 and cfrc.size
        else np.zeros(0, dtype=float)
    )
    load_total = float(np.sum(_fixed_width(load_norm, 8)))
    height = float(qpos[1]) if qpos.size > 1 else 0.0
    next_height = float(next_qpos[1]) if next_qpos.size > 1 else height
    pitch = float(qpos[2]) if qpos.size > 2 else 0.0
    next_pitch = float(next_qpos[2]) if next_qpos.size > 2 else pitch
    vertical_velocity = float(qvel[1]) if qvel.size > 1 else 0.0
    action_norm = float(np.linalg.norm(action))
    qacc_norm = float(np.linalg.norm(qacc))
    bias_norm = float(np.linalg.norm(qfrc_bias))
    passive_norm = float(np.linalg.norm(qfrc_passive))
    constraint_norm = float(np.linalg.norm(qfrc_constraint))
    actuator_norm = float(np.linalg.norm(qfrc_actuator))
    load_denom = load_total + 5e-2
    qacc_denom = qacc_norm + 5e-2
    height_denom = abs(height) + 5e-2
    action_denom = action_norm + 5e-2
    bias_qacc_alignment = float(
        np.dot(qfrc_bias, qacc) / ((bias_norm * qacc_norm) + 5e-2)
    )
    actuator_qacc_alignment = float(
        np.dot(qfrc_actuator, qacc) / ((actuator_norm * qacc_norm) + 5e-2)
    )
    vertical_balance_proxy = float(
        qfrc_bias[1] + qfrc_passive[1] + qfrc_constraint[1] + qfrc_actuator[1] - qacc[1]
    )
    low_height_weight = 1.0 / (1.0 + np.exp(12.0 * (height - 0.85)))

    return np.asarray(
        [
            bias_norm,
            float(qfrc_bias[1]),
            float(qfrc_bias[2]),
            passive_norm,
            constraint_norm,
            actuator_norm,
            qacc_norm,
            float(qacc[1]),
            float(qacc[2]),
            height,
            next_height - height,
            vertical_velocity,
            pitch,
            next_pitch - pitch,
            load_total,
            load_total / height_denom,
            load_total / qacc_denom,
            bias_norm / load_denom,
            constraint_norm / load_denom,
            passive_norm / load_denom,
            bias_qacc_alignment,
            actuator_qacc_alignment,
            vertical_balance_proxy,
            float(qfrc_bias[1]) / (abs(float(qacc[1])) + 5e-2),
            load_total * low_height_weight,
            abs(next_height - height) / action_denom,
        ],
        dtype=float,
    )


def _gravity_balance_probe_feature_vector(transition: _Transition, env) -> np.ndarray:
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    next_qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
    qfrc_bias = _fixed_width(_mujoco_data_array(env, "qfrc_bias").reshape(-1), 9)
    qfrc_passive = _fixed_width(_mujoco_data_array(env, "qfrc_passive").reshape(-1), 9)
    qfrc_constraint = _fixed_width(_mujoco_data_array(env, "qfrc_constraint").reshape(-1), 9)
    qfrc_actuator = _fixed_width(_mujoco_data_array(env, "qfrc_actuator").reshape(-1), 9)
    qacc = _fixed_width(_mujoco_data_array(env, "qacc").reshape(-1), 9)
    cfrc = _mujoco_data_array(env, "cfrc_ext")
    load_norm = (
        np.linalg.norm(cfrc.reshape(cfrc.shape[0], -1), axis=1)
        if cfrc.ndim >= 2 and cfrc.size
        else np.zeros(0, dtype=float)
    )
    load_fixed = _fixed_width(load_norm, 8)
    load_total = float(np.sum(load_fixed))
    qacc_norm = float(np.linalg.norm(qacc))
    bias_norm = float(np.linalg.norm(qfrc_bias))
    actuator_norm = float(np.linalg.norm(qfrc_actuator))
    height = float(qpos[1]) if qpos.size > 1 else 0.0
    next_height = float(next_qpos[1]) if next_qpos.size > 1 else height
    pitch = float(qpos[2]) if qpos.size > 2 else 0.0
    next_pitch = float(next_qpos[2]) if next_qpos.size > 2 else pitch
    vertical_velocity = float(qvel[1]) if qvel.size > 1 else 0.0
    action_norm = float(np.linalg.norm(action))
    load_entropy = 0.0
    if load_total > 0.0:
        probabilities = load_fixed / max(load_total, 1e-12)
        positive = probabilities[probabilities > 0.0]
        if positive.size:
            load_entropy = float(-np.sum(positive * np.log(positive)) / np.log(load_fixed.size))
    load_concentration = float(np.max(load_fixed / (load_total + 5e-2)))
    total_vertical = float(
        qfrc_bias[1] + qfrc_passive[1] + qfrc_constraint[1] + qfrc_actuator[1] - qacc[1]
    )
    bias_qacc_alignment = float(
        np.dot(qfrc_bias, qacc) / ((bias_norm * qacc_norm) + 5e-2)
    )
    actuator_qacc_alignment = float(
        np.dot(qfrc_actuator, qacc) / ((actuator_norm * qacc_norm) + 5e-2)
    )
    low_height_weight = 1.0 / (1.0 + np.exp(12.0 * (height - 0.85)))
    return np.asarray(
        [
            np.log1p(load_total),
            np.log1p(load_total / (abs(height) + 5e-2)),
            np.log1p(load_total / (qacc_norm + 5e-2)),
            _signed_log_ratio(float(qfrc_bias[1]), load_total),
            _signed_log_ratio(float(qfrc_passive[1]), load_total),
            _signed_log_ratio(float(qfrc_constraint[1]), load_total),
            _signed_log_ratio(float(qfrc_actuator[1]), load_total),
            _signed_log_ratio(float(qfrc_bias[1]), float(qacc[1])),
            _signed_log_ratio(total_vertical, load_total),
            _signed_log_ratio(total_vertical, float(qacc[1])),
            bias_qacc_alignment,
            actuator_qacc_alignment,
            _signed_log_ratio(next_height - height, action_norm),
            vertical_velocity,
            _signed_log_ratio(next_pitch - pitch, action_norm),
            np.log1p(load_total) * low_height_weight,
            load_entropy,
            load_concentration,
        ],
        dtype=float,
    )


def _walker_gravity_response_probe_target_vector(transition: _Transition, env) -> np.ndarray:
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    next_qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    next_qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
    qvel_delta = _fixed_width(next_qvel - qvel, 9)
    qfrc_bias = _fixed_width(_mujoco_data_array(env, "qfrc_bias").reshape(-1), 9)
    qfrc_passive = _fixed_width(_mujoco_data_array(env, "qfrc_passive").reshape(-1), 9)
    qfrc_constraint = _fixed_width(_mujoco_data_array(env, "qfrc_constraint").reshape(-1), 9)
    qfrc_actuator = _fixed_width(_mujoco_data_array(env, "qfrc_actuator").reshape(-1), 9)
    qacc = _fixed_width(_mujoco_data_array(env, "qacc").reshape(-1), 9)

    cfrc = _mujoco_data_array(env, "cfrc_ext")
    load_norm = (
        np.linalg.norm(cfrc.reshape(cfrc.shape[0], -1), axis=1)
        if cfrc.ndim >= 2 and cfrc.size
        else np.zeros(0, dtype=float)
    )
    load_fixed = _fixed_width(load_norm, 8)
    load_total = float(np.sum(load_fixed))
    load_share = load_fixed / (load_total + 5e-2)
    load_entropy = 0.0
    if load_total > 0.0:
        probabilities = load_fixed / max(load_total, 1e-12)
        positive = probabilities[probabilities > 0.0]
        if positive.size:
            load_entropy = float(-np.sum(positive * np.log(positive)) / np.log(load_fixed.size))
    load_concentration = float(np.max(load_share)) if load_share.size else 0.0

    xpos = _mujoco_data_array(env, "xpos")
    body_z = xpos[:, 2] if xpos.ndim == 2 and xpos.shape[1] >= 3 else np.zeros(0, dtype=float)
    body_z_fixed = _fixed_width(body_z, 8)
    low_height_weight = 1.0 / (1.0 + np.exp(12.0 * (body_z_fixed - 0.45)))
    low_height_load = float(np.sum(load_fixed * low_height_weight))
    cvel = _mujoco_data_array(env, "cvel")
    if cvel.ndim == 2 and cvel.shape[1] >= 6:
        linear_velocity = cvel[:, 3:6]
    elif cvel.ndim == 2 and cvel.shape[1] >= 3:
        linear_velocity = cvel[:, -3:]
    else:
        linear_velocity = np.zeros((0, 3), dtype=float)
    vertical_body_speed = (
        _fixed_width(linear_velocity[:, 2], 8)
        if linear_velocity.ndim == 2 and linear_velocity.shape[1] >= 3
        else np.zeros(8, dtype=float)
    )
    loaded_vertical_speed = float(np.sum(np.abs(vertical_body_speed) * np.log1p(load_fixed)))
    low_height_vertical_speed = float(np.sum(np.abs(vertical_body_speed) * low_height_weight))

    height = float(qpos[1]) if qpos.size > 1 else 0.0
    next_height = float(next_qpos[1]) if next_qpos.size > 1 else height
    pitch = float(qpos[2]) if qpos.size > 2 else 0.0
    next_pitch = float(next_qpos[2]) if next_qpos.size > 2 else pitch
    vertical_velocity = float(qvel[1]) if qvel.size > 1 else 0.0
    pitch_rate = float(qvel[2]) if qvel.size > 2 else 0.0
    action_norm = float(np.linalg.norm(action))
    bias_norm = float(np.linalg.norm(qfrc_bias))
    passive_norm = float(np.linalg.norm(qfrc_passive))
    qacc_norm = float(np.linalg.norm(qacc))
    total_vertical = float(
        qfrc_bias[1] + qfrc_passive[1] + qfrc_constraint[1] + qfrc_actuator[1] - qacc[1]
    )
    bias_qacc_alignment = float(
        np.dot(qfrc_bias, qacc) / ((bias_norm * qacc_norm) + 5e-2)
    )
    bias_passive_alignment = float(
        np.dot(qfrc_bias, qfrc_passive) / ((bias_norm * passive_norm) + 5e-2)
    )

    return np.asarray(
        [
            _signed_log_ratio(float(qfrc_bias[1]), 1.0, eps=0.0),
            _signed_log_ratio(float(qfrc_bias[2]), 1.0, eps=0.0),
            np.log1p(float(np.linalg.norm(qfrc_bias[:3]))),
            np.log1p(float(np.linalg.norm(qfrc_bias[-6:]))),
            _signed_log_ratio(bias_norm, height),
            _signed_log_ratio(bias_norm, load_total),
            _signed_log_ratio(float(qacc[1]), 1.0, eps=0.0),
            _signed_log_ratio(float(qacc[2]), 1.0, eps=0.0),
            _signed_log_ratio(total_vertical, 1.0, eps=0.0),
            _signed_log_ratio(total_vertical, load_total),
            _signed_log_ratio(next_height - height, action_norm),
            _signed_log_ratio(next_pitch - pitch, action_norm),
            vertical_velocity,
            pitch_rate,
            np.log1p(load_total),
            np.log1p(load_total / (abs(height) + 5e-2)),
            np.log1p(low_height_load),
            load_concentration,
            load_entropy,
            np.log1p(loaded_vertical_speed),
            np.log1p(low_height_vertical_speed),
            bias_qacc_alignment,
            bias_passive_alignment,
            _signed_log_ratio(float(qvel_delta[1]) if qvel_delta.size > 1 else 0.0, action_norm),
        ],
        dtype=float,
    )


def _walker_excitation_response_probe_target_vector(transition: _Transition) -> np.ndarray:
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    next_qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    next_qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
    previous_action = _previous_action_vector(transition, action.size)
    action_delta = action - previous_action
    innovation_norm = float(np.linalg.norm(action_delta))
    qvel_delta = _fixed_width(next_qvel - qvel, 9)
    qpos_delta = _fixed_width(next_qpos - qpos, 9)
    qpos_delta_rel = qpos_delta[1:] if qpos_delta.size > 1 else qpos_delta
    joint_delta = _actuated_qvel(qvel_delta, action.size)
    leg_a_delta = joint_delta[:3]
    leg_b_delta = joint_delta[3:6]
    alignment = float(np.dot(action_delta, joint_delta))
    denom = innovation_norm + 5e-2
    root_forward_delta = float(qvel_delta[0]) if qvel_delta.size > 0 else 0.0
    root_vertical_delta = float(qvel_delta[1]) if qvel_delta.size > 1 else 0.0
    root_pitch_delta = float(qvel_delta[2]) if qvel_delta.size > 2 else 0.0
    height_delta = float(qpos_delta_rel[0]) if qpos_delta_rel.size > 0 else 0.0
    pitch_delta = float(qpos_delta_rel[1]) if qpos_delta_rel.size > 1 else 0.0
    return np.concatenate(
        [
            np.asarray(
                [
                    root_forward_delta,
                    root_vertical_delta,
                    root_pitch_delta,
                    height_delta,
                    pitch_delta,
                    _signed_log_ratio(root_forward_delta, innovation_norm),
                    _signed_log_ratio(root_vertical_delta, innovation_norm),
                    _signed_log_ratio(root_pitch_delta, innovation_norm),
                    _signed_log_ratio(height_delta, innovation_norm),
                    _signed_log_ratio(pitch_delta, innovation_norm),
                ],
                dtype=float,
            ),
            joint_delta,
            joint_delta / denom,
            np.asarray(
                [
                    float(np.linalg.norm(leg_a_delta) - np.linalg.norm(leg_b_delta)),
                    _signed_log_ratio(alignment, innovation_norm),
                    abs(_signed_log_ratio(alignment, innovation_norm)),
                ],
                dtype=float,
            ),
        ]
    )


def _walker_innovation_gain_probe_target_vector(transition: _Transition) -> np.ndarray:
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    next_qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    next_qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    innovation = _policy_innovation_vector(transition, 6)
    qvel_delta = _fixed_width(next_qvel - qvel, 9)
    qpos_delta = _fixed_width(next_qpos - qpos, 9)
    qpos_delta_rel = qpos_delta[1:] if qpos_delta.size > 1 else qpos_delta
    joint_delta = _actuated_qvel(qvel_delta, innovation.size)
    leg_a_delta = joint_delta[:3]
    leg_b_delta = joint_delta[3:6]
    leg_a_innovation = innovation[:3]
    leg_b_innovation = innovation[3:6]
    innovation_sum = float(np.mean(innovation))
    innovation_asym = float(np.mean(leg_a_innovation) - np.mean(leg_b_innovation))
    innovation_norm = float(np.linalg.norm(innovation))
    root_forward_delta = float(qvel_delta[0]) if qvel_delta.size > 0 else 0.0
    root_vertical_delta = float(qvel_delta[1]) if qvel_delta.size > 1 else 0.0
    root_pitch_delta = float(qvel_delta[2]) if qvel_delta.size > 2 else 0.0
    height_delta = float(qpos_delta_rel[0]) if qpos_delta_rel.size > 0 else 0.0
    pitch_delta = float(qpos_delta_rel[1]) if qpos_delta_rel.size > 1 else 0.0
    leg_delta_asym = float(np.linalg.norm(leg_a_delta) - np.linalg.norm(leg_b_delta))
    phase_terms = _FrozenGaitPhaseResponseProbe._phase_basis(transition)[24:30]
    alignment = float(np.dot(innovation, joint_delta))
    return np.concatenate(
        [
            np.asarray(
                [
                    innovation_sum * root_forward_delta,
                    innovation_sum * root_vertical_delta,
                    innovation_asym * root_pitch_delta,
                    innovation_norm * height_delta,
                    innovation_norm * pitch_delta,
                ],
                dtype=float,
            ),
            innovation * joint_delta,
            innovation_asym * phase_terms * root_pitch_delta,
            innovation_sum * phase_terms * root_vertical_delta,
            np.asarray(
                [
                    innovation_norm * leg_delta_asym,
                    alignment,
                ],
                dtype=float,
            ),
        ]
    )


def _gravity_compensation_probe_target_vector(transition: _Transition, env) -> np.ndarray:
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    qfrc_bias = _fixed_width(_mujoco_data_array(env, "qfrc_bias").reshape(-1), 9)
    qfrc_passive = _fixed_width(_mujoco_data_array(env, "qfrc_passive").reshape(-1), 9)
    qacc = _fixed_width(_mujoco_data_array(env, "qacc").reshape(-1), 9)
    bias_norm = float(np.linalg.norm(qfrc_bias))
    passive_norm = float(np.linalg.norm(qfrc_passive))
    qacc_norm = float(np.linalg.norm(qacc))
    height = float(qpos[1]) if qpos.size > 1 else 0.0
    height_denom = abs(height) + 5e-2
    bias_qacc_alignment = float(
        np.dot(qfrc_bias, qacc) / ((bias_norm * qacc_norm) + 5e-2)
    )
    bias_passive_alignment = float(
        np.dot(qfrc_bias, qfrc_passive) / ((bias_norm * passive_norm) + 5e-2)
    )
    return np.concatenate(
        [
            qfrc_bias,
            np.asarray(
                [
                    bias_norm,
                    float(qfrc_bias[1]),
                    float(qfrc_bias[2]),
                    float(np.linalg.norm(qfrc_bias[:3])),
                    float(np.linalg.norm(qfrc_bias[-6:])),
                    bias_norm / height_denom,
                    float(qfrc_bias[1]) / height_denom,
                    bias_qacc_alignment,
                    bias_passive_alignment,
                ],
                dtype=float,
            ),
        ]
    )


def _mechanical_energy_probe_feature_vector(transition: _Transition) -> np.ndarray:
    qpos = (
        np.asarray(transition.qpos, dtype=float).reshape(-1)
        if transition.qpos is not None
        else np.zeros(0, dtype=float)
    )
    next_qpos = (
        np.asarray(transition.next_qpos, dtype=float).reshape(-1)
        if transition.next_qpos is not None
        else np.zeros(0, dtype=float)
    )
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    next_qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
    qvel_delta = _fixed_width(next_qvel - qvel, 9)
    actuated_qvel = _actuated_qvel(qvel, action.size)
    next_actuated_qvel = _actuated_qvel(next_qvel, action.size)
    actuated_delta = _actuated_qvel(qvel_delta, action.size)
    physics = (
        np.asarray(transition.physics, dtype=float).reshape(-1)
        if transition.physics is not None
        else np.zeros(len(PHYSICS_RESIDUAL_PROBE_FEATURE_NAMES), dtype=float)
    )
    qfrc_constraint = _fixed_width(physics[8:17], 9)
    qfrc_passive = _fixed_width(physics[17:26], 9)
    qacc = _fixed_width(physics[26:35], 9)
    action_norm = float(np.linalg.norm(action))
    denom = action_norm + 5e-2
    qvel_match = _fixed_width(qvel, 9)

    kinetic = 0.5 * float(np.dot(qvel, qvel)) if qvel.size else 0.0
    next_kinetic = 0.5 * float(np.dot(next_qvel, next_qvel)) if next_qvel.size else 0.0
    actuated_kinetic = 0.5 * float(np.dot(actuated_qvel, actuated_qvel))
    next_actuated_kinetic = 0.5 * float(np.dot(next_actuated_qvel, next_actuated_qvel))
    action_power = float(np.dot(action, actuated_qvel))
    action_accel_alignment = float(np.dot(action, actuated_delta) / denom)
    passive_power = float(np.dot(qfrc_passive, qvel_match))
    constraint_power = float(np.dot(qfrc_constraint, qvel_match))
    height = float(qpos[1]) if qpos.size > 1 else 0.0
    next_height = float(next_qpos[1]) if next_qpos.size > 1 else height
    body_angle = float(qpos[2]) if qpos.size > 2 else 0.0
    next_body_angle = float(next_qpos[2]) if next_qpos.size > 2 else body_angle

    return np.asarray(
        [
            kinetic,
            next_kinetic - kinetic,
            actuated_kinetic,
            next_actuated_kinetic - actuated_kinetic,
            next_height,
            next_height - height,
            next_body_angle,
            next_body_angle - body_angle,
            action_norm,
            action_power,
            action_accel_alignment,
            passive_power,
            constraint_power,
            float(np.linalg.norm(qacc)),
            _numeric_info_value(transition.info, "x_velocity"),
            _numeric_info_value(transition.info, "x_velocity") / denom,
            _numeric_info_value(transition.info, "reward_ctrl") / denom,
            float(transition.reward) / denom,
        ],
        dtype=float,
    )


def _actuator_efficiency_probe_feature_vector(transition: _Transition) -> np.ndarray:
    action = _fixed_width(np.asarray(transition.action, dtype=float), 6)
    qvel = (
        np.asarray(transition.qvel, dtype=float).reshape(-1)
        if transition.qvel is not None
        else np.zeros(0, dtype=float)
    )
    next_qvel = (
        np.asarray(transition.next_qvel, dtype=float).reshape(-1)
        if transition.next_qvel is not None
        else np.zeros(0, dtype=float)
    )
    qvel_delta = _fixed_width(next_qvel - qvel, 9)
    actuated_qvel = _actuated_qvel(qvel, action.size)
    actuated_delta = _actuated_qvel(qvel_delta, action.size)
    action_norm = float(np.linalg.norm(action))
    delta_norm = float(np.linalg.norm(actuated_delta))
    denom = action_norm + 5e-2
    projected = float(np.dot(action, actuated_delta) / (np.dot(action, action) + 5e-2))
    alignment = projected / (delta_norm + 5e-2)

    qpos_delta_rel = np.zeros(0, dtype=float)
    if transition.qpos is not None and transition.next_qpos is not None:
        qpos_delta = np.asarray(transition.next_qpos, dtype=float) - np.asarray(
            transition.qpos,
            dtype=float,
        )
        qpos_delta_rel = qpos_delta[1:] if qpos_delta.size > 1 else qpos_delta

    return np.asarray(
        [
            action_norm,
            float(np.linalg.norm(actuated_qvel)),
            delta_norm,
            delta_norm / denom,
            projected,
            abs(projected),
            alignment,
            _numeric_info_value(transition.info, "x_velocity") / denom,
            _numeric_info_value(transition.info, "reward_forward") / denom,
            _numeric_info_value(transition.info, "reward_ctrl") / denom,
            float(transition.reward) / denom,
            float(qpos_delta_rel[0]) if qpos_delta_rel.size > 0 else 0.0,
            float(qpos_delta_rel[1]) if qpos_delta_rel.size > 1 else 0.0,
        ],
        dtype=float,
    )


def _task_reward_probe_feature_vector(transition: _Transition) -> np.ndarray:
    return np.asarray(
        [
            float(transition.reward),
            _numeric_info_value(transition.info, "reward_dist"),
            _numeric_info_value(transition.info, "reward_near"),
            _numeric_info_value(transition.info, "reward_forward"),
        ],
        dtype=float,
    )


def _task_observation_coords(obs: np.ndarray, env_id: str) -> np.ndarray:
    values = np.asarray(obs, dtype=float).reshape(-1)
    if env_id.startswith("Reacher") and values.size >= 6:
        return values[4:6].copy()
    if env_id.startswith("Pusher") and values.size >= 22:
        return values[20:22].copy()
    return np.zeros(2, dtype=float)


def _task_observation_probe_feature_vector(
    transition: _Transition,
    env_id: str,
) -> np.ndarray:
    current = _task_observation_coords(transition.obs, env_id)
    observed = _task_observation_coords(transition.next_obs, env_id)
    return np.asarray(
        [
            float(observed[0]),
            float(observed[1]),
            float(np.linalg.norm(observed)),
            float(np.linalg.norm(observed - current)),
        ],
        dtype=float,
    )


def generate_mujoco_internal_stream(config: MujocoTraceConfig) -> SimulatedRun:
    """Roll out a Gymnasium MuJoCo env and emit CADET feature traces.

    The returned stream uses the paper's five monitored channels:
    ``td_abs, value, entropy, max_logit, hidden_norm``. The PUR deltas are
    computed by evaluating the online actor-value update on a frozen transition
    buffer before and after each parameter update.
    """

    gym = _require_gymnasium()
    env = gym.make(config.env_id)
    probe_env = None
    if config.feature_mode in {"actor_value_reset_probe", "reset_probe"}:
        probe_env = gym.make(config.env_id)
        probe_env.reset(seed=config.seed + 7919)
    try:
        obs, _ = env.reset(seed=config.seed)
        frozen_task_target = _get_task_target(env) if config.freeze_task_goal else None
        frozen_probe_target = None
        if config.freeze_task_goal and probe_env is not None:
            frozen_probe_target = _get_task_target(probe_env)
        env.action_space.seed(config.seed)
        action_low = np.asarray(env.action_space.low, dtype=float)
        action_high = np.asarray(env.action_space.high, dtype=float)
        finite_low = np.where(np.isfinite(action_low), action_low, -1.0)
        finite_high = np.where(np.isfinite(action_high), action_high, 1.0)
        agent = _LinearActorValue(
            obs_dim=int(np.asarray(obs).size),
            action_dim=int(np.asarray(action_low).size),
            hidden_dim=config.hidden_dim,
            action_low=finite_low,
            action_high=finite_high,
            policy_std=config.policy_std,
            seed=config.seed + 17,
        )

        tau = None if config.shift == "no_shift" else config.tau
        sensor_bias = (
            _sensor_bias_vector(np.asarray(obs).size, config.sensor_bias_scale, config.seed)
            if config.shift == "sensor_bias"
            else np.zeros(0, dtype=float)
        )
        features: list[np.ndarray] = []
        deltas: list[np.ndarray] = []
        update_diagnostics: list[dict[str, float]] = []
        frozen_buffer: list[_Transition] = []
        pending_buffer: list[_Transition] = []
        shifted = False
        episode_step = 0
        previous_action = np.zeros_like(finite_low, dtype=float)
        previous_policy_innovation = np.zeros_like(finite_low, dtype=float)
        transition_probe = (
            _FrozenLinearTransitionProbe()
            if config.feature_mode
            in {"model_probe", "model_state_probe", "model_vector_probe", "model_component_probe"}
            else None
        )
        knn_transition_probe = (
            _FrozenKNNTransitionProbe(config.knn_neighbors)
            if config.feature_mode == "knn_state_probe"
            else None
        )
        phase_transition_probe = (
            _FrozenPhaseTransitionProbe(config.knn_neighbors)
            if config.feature_mode == "phase_state_probe"
            else None
        )
        physics_residual_probe = (
            _FrozenPhysicsResidualProbe()
            if config.feature_mode == "physics_residual_probe"
            else None
        )
        force_balance_probe = (
            _FrozenForceBalanceResidualProbe()
            if config.feature_mode == "force_balance_probe"
            else None
        )
        gravity_residual_probe = (
            _FrozenGravityResidualProbe()
            if config.feature_mode == "gravity_residual_probe"
            else None
        )
        gravity_compensation_probe = (
            _FrozenGravityCompensationProbe()
            if config.feature_mode == "gravity_compensation_probe"
            else None
        )
        walker_gravity_response_probe = (
            _FrozenWalkerGravityResponseProbe()
            if config.feature_mode == "walker_gravity_response_probe"
            else None
        )
        walker_excitation_response_probe = (
            _FrozenWalkerExcitationResponseProbe()
            if config.feature_mode == "walker_excitation_response_probe"
            else None
        )
        walker_innovation_gain_probe = (
            _FrozenWalkerInnovationGainProbe()
            if config.feature_mode == "walker_innovation_gain_probe"
            else None
        )
        support_response_probe = (
            _FrozenSupportResponseProbe()
            if config.feature_mode == "support_response_probe"
            else None
        )
        kernel_support_response_probe = (
            _FrozenKernelSupportResponseProbe()
            if config.feature_mode == "kernel_support_response_probe"
            else None
        )
        control_response_probe = (
            _FrozenControlResponseProbe()
            if config.feature_mode == "control_response_probe"
            else None
        )
        gait_phase_response_probe = (
            _FrozenGaitPhaseResponseProbe()
            if config.feature_mode == "gait_phase_response_probe"
            else None
        )
        gait_load_response_probe = (
            _FrozenGaitLoadResponseProbe()
            if config.feature_mode == "gait_load_response_probe"
            else None
        )
        sensor_consistency_probe = (
            _FrozenSensorConsistencyProbe()
            if config.feature_mode == "sensor_consistency_probe"
            else None
        )

        for t in range(config.horizon):
            if tau is not None and t == tau and not shifted:
                if config.shift == "sensor_bias":
                    obs = _apply_sensor_bias(obs, sensor_bias)
                else:
                    _apply_shift(
                        env,
                        config.shift,
                        config.gravity_scale,
                        config.friction_scale,
                        config.target_shift,
                    )
                if config.freeze_task_goal:
                    frozen_task_target = _get_task_target(env)
                if probe_env is not None and config.shift != "sensor_bias":
                    _apply_shift(
                        probe_env,
                        config.shift,
                        config.gravity_scale,
                        config.friction_scale,
                        config.target_shift,
                    )
                    if config.freeze_task_goal:
                        frozen_probe_target = _get_task_target(probe_env)
                shifted = True

            policy_mean = np.asarray(agent.policy_mean(obs), dtype=float)
            if config.policy_mode == "linear_actor":
                action = agent.act(obs)
            elif config.policy_mode == "linear_sinusoidal_dither":
                action = _linear_sinusoidal_dither_action(
                    agent,
                    np.asarray(obs, dtype=float),
                    t=t,
                    action_low=finite_low,
                    action_high=finite_high,
                    env_id=config.env_id,
                )
            elif config.policy_mode == "linear_probe_burst":
                action = _linear_probe_burst_action(
                    agent,
                    np.asarray(obs, dtype=float),
                    t=t,
                    action_low=finite_low,
                    action_high=finite_high,
                    env_id=config.env_id,
                )
            else:
                action = _scripted_policy_action(
                    config.policy_mode,
                    t=t,
                    action_low=finite_low,
                    action_high=finite_high,
                    env_id=config.env_id,
                )
            policy_innovation = (
                (np.asarray(action, dtype=float) - policy_mean)
                / max(abs(config.policy_std), 1e-8)
            )
            qpos, qvel = _simulator_state(env)
            executed_action = (
                _apply_actuator_loss(action, config.actuator_loss_scale)
                if config.shift == "actuator_loss" and shifted
                else action
            )
            next_obs, reward, terminated, truncated, info = env.step(executed_action)
            if config.shift == "sensor_bias" and shifted:
                next_obs = _apply_sensor_bias(next_obs, sensor_bias)
            next_qpos, next_qvel = _simulator_state(env)
            if config.feature_mode == "force_balance_probe":
                physics = _force_balance_probe_target_vector(env)
            elif config.feature_mode == "gravity_residual_probe":
                physics = None
            elif config.feature_mode == "support_response_probe":
                physics = None
            else:
                physics = _physics_probe_target_vector(env)
            done = bool(terminated or truncated)
            transition = _Transition(
                obs=np.asarray(obs, dtype=float).copy(),
                action=np.asarray(action, dtype=float).copy(),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=float).copy(),
                done=done,
                info={
                    **dict(info),
                    "commanded_action": np.asarray(action, dtype=float).copy(),
                    "executed_action": np.asarray(executed_action, dtype=float).copy(),
                    "previous_action": np.asarray(previous_action, dtype=float).copy(),
                    "policy_mean": policy_mean.copy(),
                    "policy_std": float(config.policy_std),
                    "policy_innovation": np.asarray(policy_innovation, dtype=float).copy(),
                    "previous_policy_innovation": np.asarray(
                        previous_policy_innovation,
                        dtype=float,
                    ).copy(),
                },
                qpos=qpos,
                qvel=qvel,
                next_qpos=next_qpos,
                next_qvel=next_qvel,
                physics=physics,
            )
            if config.feature_mode == "gravity_residual_probe":
                transition.physics = _gravity_signature_probe_feature_vector(transition, env)
            if config.feature_mode == "gravity_compensation_probe":
                transition.physics = _gravity_compensation_probe_target_vector(transition, env)
            if config.feature_mode == "walker_gravity_response_probe":
                transition.physics = _walker_gravity_response_probe_target_vector(transition, env)
            if config.feature_mode == "support_response_probe":
                transition.physics = _support_response_probe_target_vector(transition, env)
            if config.feature_mode == "kernel_support_response_probe":
                transition.physics = _support_response_probe_target_vector(transition, env)
            if config.feature_mode == "gait_load_response_probe":
                transition.physics = _support_response_probe_target_vector(transition, env)
            if (
                transition_probe is not None
                and not transition_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                transition_probe.fit(frozen_buffer)
            if (
                knn_transition_probe is not None
                and not knn_transition_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                knn_transition_probe.fit(frozen_buffer)
            if (
                phase_transition_probe is not None
                and not phase_transition_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                phase_transition_probe.fit(frozen_buffer)
            if (
                physics_residual_probe is not None
                and not physics_residual_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                physics_residual_probe.fit(frozen_buffer)
            if (
                force_balance_probe is not None
                and not force_balance_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                force_balance_probe.fit(frozen_buffer)
            if (
                gravity_residual_probe is not None
                and not gravity_residual_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                gravity_residual_probe.fit(frozen_buffer)
            if (
                gravity_compensation_probe is not None
                and not gravity_compensation_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                gravity_compensation_probe.fit(frozen_buffer)
            if (
                walker_gravity_response_probe is not None
                and not walker_gravity_response_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                walker_gravity_response_probe.fit(frozen_buffer)
            if (
                walker_excitation_response_probe is not None
                and not walker_excitation_response_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                walker_excitation_response_probe.fit(frozen_buffer)
            if (
                walker_innovation_gain_probe is not None
                and not walker_innovation_gain_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                walker_innovation_gain_probe.fit(frozen_buffer)
            if (
                support_response_probe is not None
                and not support_response_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                support_response_probe.fit(frozen_buffer)
            if (
                kernel_support_response_probe is not None
                and not kernel_support_response_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                kernel_support_response_probe.fit(frozen_buffer)
            if (
                control_response_probe is not None
                and not control_response_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                control_response_probe.fit(frozen_buffer)
            if (
                gait_phase_response_probe is not None
                and not gait_phase_response_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                gait_phase_response_probe.fit(frozen_buffer)
            if (
                gait_load_response_probe is not None
                and not gait_load_response_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                gait_load_response_probe.fit(frozen_buffer)
            if (
                sensor_consistency_probe is not None
                and not sensor_consistency_probe.fitted
                and len(frozen_buffer) >= config.frozen_buffer_size
            ):
                sensor_consistency_probe.fit(frozen_buffer)

            actor_features = agent.feature_vector(transition, config.gamma)
            if config.feature_mode == "actor_value":
                z_t = actor_features
            elif config.feature_mode == "actor_value_kinematic":
                z_t = np.concatenate(
                    [actor_features, _kinematic_feature_vector(transition, env)]
                )
            elif config.feature_mode == "actor_value_telemetry":
                z_t = np.concatenate(
                    [
                        actor_features,
                        _kinematic_feature_vector(transition, env),
                        _telemetry_feature_vector(transition, TELEMETRY_FEATURE_NAMES),
                    ]
                )
            elif config.feature_mode == "actor_value_domain":
                z_t = np.concatenate(
                    [
                        actor_features,
                        _kinematic_feature_vector(transition, env),
                        _telemetry_feature_vector(transition, DOMAIN_FEATURE_NAMES),
                    ]
                )
            elif config.feature_mode == "actor_value_reset_probe":
                z_t = np.concatenate(
                    [
                        actor_features,
                        _kinematic_feature_vector(transition, env),
                        _telemetry_feature_vector(transition, DOMAIN_FEATURE_NAMES),
                        _reset_probe_feature_vector(frozen_buffer, probe_env),
                    ]
                )
            elif config.feature_mode == "reset_probe":
                z_t = _reset_probe_feature_vector(frozen_buffer, probe_env)
            elif config.feature_mode == "model_probe":
                assert transition_probe is not None
                z_t = transition_probe.feature_vector(transition)
            elif config.feature_mode == "model_state_probe":
                assert transition_probe is not None
                z_t = transition_probe.feature_vector(transition)[1:]
            elif config.feature_mode == "model_vector_probe":
                assert transition_probe is not None
                z_t = transition_probe.signed_projection_features(transition)
            elif config.feature_mode == "model_component_probe":
                assert transition_probe is not None
                z_t = transition_probe.component_features(transition)
            elif config.feature_mode == "motion_probe":
                z_t = _motion_probe_feature_vector(transition)
            elif config.feature_mode == "locomotion_observation_probe":
                z_t = _locomotion_observation_probe_feature_vector(transition)
            elif config.feature_mode == "locomotion_dynamics_probe":
                z_t = _locomotion_dynamics_probe_feature_vector(transition, env)
            elif config.feature_mode == "locomotion_stability_probe":
                z_t = _locomotion_stability_probe_feature_vector(transition, episode_step)
            elif config.feature_mode == "gravity_signature_probe":
                z_t = _gravity_signature_probe_feature_vector(transition, env)
            elif config.feature_mode == "gravity_residual_probe":
                assert gravity_residual_probe is not None
                z_t = gravity_residual_probe.feature_vector(transition)
            elif config.feature_mode == "gravity_compensation_probe":
                assert gravity_compensation_probe is not None
                z_t = gravity_compensation_probe.feature_vector(transition)
            elif config.feature_mode == "gravity_balance_probe":
                z_t = _gravity_balance_probe_feature_vector(transition, env)
            elif config.feature_mode == "walker_gravity_response_probe":
                assert walker_gravity_response_probe is not None
                z_t = walker_gravity_response_probe.feature_vector(transition)
            elif config.feature_mode == "walker_excitation_response_probe":
                assert walker_excitation_response_probe is not None
                z_t = walker_excitation_response_probe.feature_vector(transition)
            elif config.feature_mode == "walker_innovation_gain_probe":
                assert walker_innovation_gain_probe is not None
                z_t = walker_innovation_gain_probe.feature_vector(transition)
            elif config.feature_mode == "mechanical_energy_probe":
                z_t = _mechanical_energy_probe_feature_vector(transition)
            elif config.feature_mode == "control_response_probe":
                assert control_response_probe is not None
                z_t = control_response_probe.feature_vector(transition)
            elif config.feature_mode == "gait_phase_response_probe":
                assert gait_phase_response_probe is not None
                z_t = gait_phase_response_probe.feature_vector(transition)
            elif config.feature_mode == "gait_load_response_probe":
                assert gait_load_response_probe is not None
                z_t = gait_load_response_probe.feature_vector(transition)
            elif config.feature_mode == "actuator_efficiency_probe":
                z_t = _actuator_efficiency_probe_feature_vector(transition)
            elif config.feature_mode == "sensor_consistency_probe":
                assert sensor_consistency_probe is not None
                z_t = sensor_consistency_probe.feature_vector(transition)
            elif config.feature_mode == "task_reward_probe":
                z_t = _task_reward_probe_feature_vector(transition)
            elif config.feature_mode == "task_observation_probe":
                z_t = _task_observation_probe_feature_vector(transition, config.env_id)
            elif config.feature_mode == "knn_state_probe":
                assert knn_transition_probe is not None
                z_t = knn_transition_probe.feature_vector(transition)
            elif config.feature_mode == "phase_state_probe":
                assert phase_transition_probe is not None
                z_t = phase_transition_probe.feature_vector(transition)
            elif config.feature_mode == "physics_residual_probe":
                assert physics_residual_probe is not None
                z_t = physics_residual_probe.feature_vector(transition)
            elif config.feature_mode == "force_balance_probe":
                assert force_balance_probe is not None
                z_t = force_balance_probe.feature_vector(transition)
            elif config.feature_mode == "contact_slip_probe":
                z_t = _contact_slip_probe_feature_vector(env)
            elif config.feature_mode == "support_telemetry_probe":
                z_t = _support_response_probe_target_vector(transition, env)
            elif config.feature_mode == "support_response_probe":
                assert support_response_probe is not None
                z_t = support_response_probe.feature_vector(transition)
            elif config.feature_mode == "kernel_support_response_probe":
                assert kernel_support_response_probe is not None
                z_t = kernel_support_response_probe.feature_vector(transition)
            else:  # Defensive; __post_init__ normally catches this.
                raise ValueError(f"unknown MuJoCo feature mode {config.feature_mode!r}")

            if config.probe_update_mode == "online":
                before = agent.copy()
                if frozen_buffer:
                    mu_before = before.mean_features(frozen_buffer, config.gamma)
                else:
                    mu_before = before.feature_vector(transition, config.gamma)
                agent.update(transition, config.gamma, config.policy_lr, config.value_lr)
                if frozen_buffer:
                    mu_after = agent.mean_features(frozen_buffer, config.gamma)
                else:
                    mu_after = agent.feature_vector(transition, config.gamma)
                actor_delta = mu_after - mu_before
                if config.record_update_diagnostics and frozen_buffer:
                    update_diagnostics.append(
                        _pur_update_diagnostics(
                            before,
                            agent,
                            frozen_buffer,
                            config.gamma,
                            config.jvp_probe_eps,
                            step=t,
                            tau=tau,
                        )
                    )
            elif config.probe_update_mode == "frozen_policy":
                actor_delta = np.zeros(len(DEFAULT_FEATURE_NAMES), dtype=float)
            else:  # Defensive; __post_init__ normally catches this.
                raise ValueError(
                    f"unknown MuJoCo probe update mode {config.probe_update_mode!r}"
                )

            features.append(z_t)
            if config.feature_mode == "actor_value":
                deltas.append(actor_delta)
            elif config.feature_mode in {
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
                "task_reward_probe",
                "task_observation_probe",
                "knn_state_probe",
                "phase_state_probe",
                "physics_residual_probe",
                "force_balance_probe",
                "contact_slip_probe",
                "support_telemetry_probe",
                "support_response_probe",
                "kernel_support_response_probe",
            }:
                deltas.append(np.zeros(len(z_t), dtype=float))
            else:
                deltas.append(
                    np.concatenate(
                        [
                            actor_delta,
                            np.zeros(len(z_t) - len(DEFAULT_FEATURE_NAMES), dtype=float),
                        ]
                    )
                )

            if len(frozen_buffer) < config.frozen_buffer_size:
                frozen_buffer.append(transition)
            elif config.frozen_refresh:
                pending_buffer.append(transition)
                if len(pending_buffer) >= config.frozen_refresh:
                    frozen_buffer = pending_buffer[-config.frozen_buffer_size :]
                    pending_buffer = []

            if done:
                obs, _ = env.reset()
                episode_step = 0
                previous_action = np.zeros_like(action, dtype=float)
                previous_policy_innovation = np.zeros_like(action, dtype=float)
                if config.freeze_task_goal and frozen_task_target is not None:
                    reset_obs = _set_task_target(env, frozen_task_target)
                    if reset_obs is not None:
                        obs = reset_obs
                if config.shift == "sensor_bias" and shifted:
                    obs = _apply_sensor_bias(obs, sensor_bias)
                if (
                    config.freeze_task_goal
                    and probe_env is not None
                    and frozen_probe_target is not None
                ):
                    _set_task_target(probe_env, frozen_probe_target)
            else:
                obs = next_obs
                episode_step += 1
                previous_action = np.asarray(action, dtype=float).copy()
                previous_policy_innovation = np.asarray(policy_innovation, dtype=float).copy()

        run = SimulatedRun(
            z=np.asarray(features, dtype=float),
            endogenous_delta=np.asarray(deltas, dtype=float),
            tau=tau,
            feature_names=mujoco_feature_names(config.feature_mode),
            scenario=f"mujoco_{config.env_id}_{config.shift}",
        )
        if config.record_update_diagnostics:
            run.update_diagnostics = update_diagnostics
        return run
    finally:
        env.close()
        if probe_env is not None:
            probe_env.close()
