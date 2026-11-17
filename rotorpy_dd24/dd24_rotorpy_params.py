"""RotorPy vehicle params for the DuckieDrone DD24.

Same schema as rotorpy/vehicles/crazyflie_params.py, but with DD24 numbers.
Mass and arm length are pulled from ../environments/dd24_params.py so there is
ONE source of truth. Rotor/inertia coefficients are derived or estimated; every
estimate is tagged and carries the formula to replace it with a measurement.

RotorPy is what the DuckieDrone team actually uses for SIL (PX4 + rotorpy), so
this is the faithful simulator for the DD24's real flight controller.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.dd24_params import DD24

g = 9.81

# --- Geometry -----------------------------------------------------------
d = DD24.arm_length_m  # [EST] center->motor; measure wheelbase/2

# --- Propulsion ---------------------------------------------------------
# rotor_speed_max: DX2205 2300KV on 4S (14.8V) ~ 2300*14.8 = 34040 rpm no-load
# = 3564 rad/s; under load ~80%. [EST] until logged.
rotor_speed_max = 3000.0  # rad/s [EST]

# k_eta (thrust coeff, N/(rad/s)^2) chosen so max thrust/motor matches
# dd24_params at rotor_speed_max:  thrust = k_eta * omega^2.
# Replace by fitting thrust-stand data: k_eta = thrust / omega^2.
k_eta = DD24.max_thrust_per_motor_N / (rotor_speed_max ** 2)  # [EST]

# k_m (yaw moment coeff) = yaw_coeff (m) * k_eta.
k_m = DD24.yaw_coeff * k_eta  # [EST] shared with the Isaac env

quad_params = {
    # Inertial properties
    'mass': DD24.mass_kg,          # [SRC] user-measured 0.80 kg
    # Inertia: single source of truth in environments/dd24_params.py
    'Ixx': DD24.Ixx,
    'Iyy': DD24.Iyy,
    'Izz': DD24.Izz,
    'Ixy': DD24.Ixy,
    'Iyz': DD24.Iyz,
    'Ixz': DD24.Ixz,

    # Geometric properties (X config, vectors relative to center of mass)
    'num_rotors': 4,
    'rotor_pos': {
        'r1': d * np.array([ 0.70710678118,  0.70710678118, 0]),
        'r2': d * np.array([ 0.70710678118, -0.70710678118, 0]),
        'r3': d * np.array([-0.70710678118, -0.70710678118, 0]),
        'r4': d * np.array([-0.70710678118,  0.70710678118, 0]),
    },
    'rotor_directions': np.array([1, -1, 1, -1]),
    'rI': np.array([0, 0, 0]),     # IMU location

    # Frame aerodynamic (parasitic) drag — [EST], small; identify from fast flight.
    'c_Dx': 0.0,
    'c_Dy': 0.0,
    'c_Dz': 0.0,

    # Rotor properties
    'k_eta': k_eta,                # [EST] from thrust stand
    'k_m':   k_m,                  # [EST] from yaw logs
    'k_d':   1.0e-6,               # [EST] rotor drag (placeholder)
    'k_z':   7.6e-7,               # [EST] induced inflow (placeholder)
    'k_h':   0.0,
    'k_flap': 0.0,

    # Motor properties
    'tau_m': 0.072,                # [EST] motor response time, s
    'rotor_speed_min': 0.0,
    'rotor_speed_max': rotor_speed_max,
    'motor_noise_std': 0.0,

    # Lower-level controller gains used by cmd_vel/cmd_ctbr/cmd_ctatt.
    # [EST] carried from Crazyflie — WILL likely need retuning for 0.8 kg.
    # Body-rate P gain for cmd_ctbr (commanded moment = I * k_w * rate_error).
    # The Crazyflie's 200 is unstable with the DD24's tau_m: a 2 rad/s step
    # overshoots ~90% and the motors saturate, adding thrust (the drone climbs).
    # 15 gives ~20% overshoot, ~140 ms rise with tau_m=0.072. Retune if tau_m changes
    # (a step-test sweep is in the Oscar README), or match PX4's MC_ROLLRATE_P.
    'k_w': 15,
    'k_v': 10,
    'kp_att': 310,
    'kd_att': 57.0,
}


if __name__ == "__main__":
    max_thrust = k_eta * rotor_speed_max ** 2
    print(f"mass            {quad_params['mass']} kg")
    print(f"arm length      {d} m")
    print(f"rotor_speed_max {rotor_speed_max} rad/s")
    print(f"k_eta           {k_eta:.3e}")
    print(f"max thrust/motor {max_thrust:.2f} N   total {4*max_thrust:.2f} N")
    print(f"thrust-to-weight {4*max_thrust/(quad_params['mass']*g):.2f}")
