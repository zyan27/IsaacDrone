"""PX4 v1.15.4 controller parameters as flown on the DD24-B.

The DD24 runs PX4 1.15.4 (diatone_mamba-f405-mk2_default.bin) with autostart
4001 (generic quad-X) and NO custom rate/attitude tuning, so its gains are the
compiled-in v1.15.4 defaults below. The only controller-relevant override in
fc_firmware/duckiedrone-px4-v4.params is IMU_DGYRO_CUTOFF (20 Hz vs 30 Hz).

Sources (tag v1.15.4):
  src/modules/mc_rate_control/mc_rate_control_params.c
  src/modules/mc_att_control/mc_att_control_params.c
  src/modules/control_allocator/module.yaml
  src/modules/sensors/vehicle_angular_velocity/imu_gyro_parameters.c
  src/modules/mc_pos_control/multicopter_position_control_params.c

Used by the batched PX4 rate-controller emulator (Part 3a) so that a policy
trained without PX4 sees the same inner loop it will fly against.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PX4RateGains:
    # Per axis (roll, pitch, yaw). Output is normalized torque in [-1, 1].
    P: tuple[float, float, float] = (0.15, 0.15, 0.2)
    I: tuple[float, float, float] = (0.2, 0.2, 0.1)
    D: tuple[float, float, float] = (0.003, 0.003, 0.0)
    FF: tuple[float, float, float] = (0.0, 0.0, 0.0)
    K: tuple[float, float, float] = (1.0, 1.0, 1.0)
    int_lim: tuple[float, float, float] = (0.30, 0.30, 0.30)  # MC_RR/PR/YR_INT_LIM


@dataclass(frozen=True)
class PX4AttGains:
    P: tuple[float, float, float] = (6.5, 6.5, 2.8)  # MC_ROLL_P, MC_PITCH_P, MC_YAW_P
    yaw_weight: float = 0.4                         # MC_YAW_WEIGHT
    rate_max_deg_s: tuple[float, float, float] = (220.0, 220.0, 200.0)  # MC_*RATE_MAX


@dataclass(frozen=True)
class PX4Filters:
    gyro_cutoff_hz: float = 40.0    # IMU_GYRO_CUTOFF (default; DD24 doesn't override)
    dgyro_cutoff_hz: float = 20.0   # IMU_DGYRO_CUTOFF — DD24 override (default 30)
    gyro_rate_max_hz: float = 800.0  # IMU_GYRO_RATEMAX
    integ_rate_hz: float = 250.0     # IMU_INTEG_RATE


@dataclass(frozen=True)
class PX4QuadX:
    # CA_ROTOR{i}_PX/PY/KM, PX4 motor numbering (front-right, rear-left,
    # front-left, rear-right). CT default 6.5 is only used for allocation scaling.
    px: tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)
    py: tuple[float, float, float, float] = (1.0, -1.0, -1.0, 1.0)
    km: tuple[float, float, float, float] = (0.05, 0.05, -0.05, -0.05)
    ct: float = 6.5
    thr_hover: float = 0.5  # MPC_THR_HOVER default; DD24 value unknown


RATE = PX4RateGains()
ATT = PX4AttGains()
FILTERS = PX4Filters()
QUAD_X = PX4QuadX()
