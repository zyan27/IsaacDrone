"""Physical parameters for the Duckietown DuckieDrone DD24 (DD24-B revision).

Single source of truth for the DD24 airframe so both the Isaac Lab RL
environment and any PX4-SITL / rotorpy model can read the same numbers.

Values are tagged:
  [SRC]  = sourced from Duckietown docs / store (see URL)
  [EST]  = engineering estimate, NOT measured — replace before trusting sim2real
  [TODO] = must be measured on the physical drone you already have

Sources:
  - Components:  https://docs.duckietown.com/daffy/opmanual-dd24/preliminaries/dd24-components-description.html
  - Store spec:  https://get.duckietown.com/products/autonomous-raspberrypi-quadcopter-duckiedrone-dd24
  - FC / PX4:    https://docs.duckietown.com/ente/opmanual-dd24/30-duckiedrone-setup-and-config/40-setting-up-the-flight-controller.html
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DD24Params:
    # --- Mass ------------------------------------------------------------
    # All-up flight weight 0.82704 kg (user-measured on a scale). The 1.76 kg
    # store/manual figure is the kit/box weight, not flight mass. Scales
    # thrust-to-weight directly, so re-confirm on a scale if the build changes.
    mass_kg: float = 0.828    # [SRC] user-measured AUW (bifilar-pendulum sheet, 828 g)

    # Component masses that are known, for a bottom-up sanity check vs the scale:
    motor_mass_kg: float = 0.028        # [SRC] DX2205, ~28 g each
    prop_mass_kg: float = 0.0035        # [SRC] Diatone 4040 3-blade, 3.5 g each
    battery_mass_kg: float = 0.178      # [SRC] 4S 1500 mAh pack, 178 g
    n_rotors: int = 4

    # --- Geometry --------------------------------------------------------
    # Arm length = frame centre to motor shaft. Measured from the DD24 CAD
    # (DD24.step/.stl): motor centres at (+-80, +-80) mm -> 113.3 mm average,
    # i.e. a ~227 mm wheelbase.
    arm_length_m: float = 0.1133        # [SRC] CAD

    # --- Propulsion ------------------------------------------------------
    # [TODO(measure)] Max thrust PER MOTOR at full throttle on a 4S pack.
    # Get it from a thrust stand or the motor/prop manufacturer chart.
    # DX2205 2300KV on 4S + 4040 is roughly ~1.0-1.2 kgf per motor.
    max_thrust_per_motor_N: float = 10.8  # [EST] ~1.1 kgf * 9.81
    kv: float = 2300.0                   # [SRC] motor KV
    battery_cells: int = 4               # [SRC] 4S
    battery_nominal_V: float = 14.8      # [SRC]

    # Yaw torque coefficient = rotor reaction torque per newton of thrust (k_m/k_eta,
    # units of metres). The old 1e-5 (copied from the Crazyflie env) is ~1000x too
    # small for a 5" prop and left the policy with essentially no yaw authority.
    # Typical 5" props are ~0.01-0.02 m. [TODO] identify from a yaw-step test.
    yaw_coeff: float = 0.0125            # [EST]

    # --- Inertia (body frame: x fwd, y left, z up), kg*m^2 ---------------------
    # Bifilar-pendulum measurement (CSCI 1951R Desmos sheet): m=0.828 kg, wire
    # half-spacing d=0.085 m, I = m g d^2 T^2 / (4 pi^2 L), T = mean of 5 runs of
    # 5 oscillations. Run-to-run scatter gives ~1-2% uncertainty on each value.
    #   roll  L=0.29 m, T=0.837 s    pitch L=0.30 m, T=0.887 s    yaw L=0.36 m, T=1.216 s
    # Used by BOTH the Isaac USD (via assets/dd24) and RotorPy.
    Ixx: float = 3.593e-3                # [SRC] measured
    Iyy: float = 3.900e-3                # [SRC] measured
    Izz: float = 6.106e-3                # [SRC] measured
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0

    # --- Flight controller ----------------------------------------------
    fc_board: str = "mamba-f405-mk2"     # [SRC] PX4 target
    fc_firmware: str = "PX4"             # [SRC] DD24-B runs PX4

    # --- Derived helpers -------------------------------------------------
    @property
    def weight_N(self) -> float:
        return self.mass_kg * 9.81

    @property
    def max_total_thrust_N(self) -> float:
        return self.max_thrust_per_motor_N * self.n_rotors

    @property
    def thrust_to_weight(self) -> float:
        """Max total thrust / weight. ~2 is sluggish, 4-6 is typical, >6 racy."""
        return self.max_total_thrust_N / self.weight_N

    def component_mass_lower_bound_kg(self) -> float:
        """Bottom-up mass of just motors+props+battery (sanity-check the scale)."""
        return (
            self.n_rotors * (self.motor_mass_kg + self.prop_mass_kg)
            + self.battery_mass_kg
        )


DD24 = DD24Params()


# --- Time-of-flight range sensors ----------------------------------------------
# Five identical modules found in the DD24 CAD (16 x 10.5 mm board, 5 x 2.6 x 1.3 mm
# chip, i.e. VL53L1X-sized). Facing = the side of the board the chip is on.
# Body frame: x forward (camera side), y left, z up; origin at the CG.
#   (name, position_m, unit_direction)
DD24_TOF_SENSORS = (
    ("down",  (-0.067,  0.000, -0.008), ( 0.0,  0.0, -1.0)),
    ("up",    (-0.067,  0.000,  0.037), ( 0.0,  0.0,  1.0)),
    ("front", ( 0.103,  0.000, -0.010), ( 1.0,  0.0,  0.0)),
    ("left",  ( 0.000,  0.044,  0.015), ( 0.0,  1.0,  0.0)),
    ("right", ( 0.000, -0.044,  0.015), ( 0.0, -1.0,  0.0)),
)
# [EST] VL53L1X: up to ~4 m in long-distance mode (less in bright light / on dark
# or sparse targets like a net), ~4 cm minimum. Noise grows with distance.
TOF_MAX_RANGE_M = 4.0
TOF_MIN_RANGE_M = 0.04
TOF_NOISE_STD_M = 0.005        # constant part
TOF_NOISE_STD_FRAC = 0.01      # plus this fraction of the distance


if __name__ == "__main__":
    p = DD24
    print(f"mass (AUW):            {p.mass_kg:.3f} kg   [user-measured]")
    print(f"inertia Ixx Iyy Izz:   {p.Ixx:.3e} {p.Iyy:.3e} {p.Izz:.3e} kg m^2  [bifilar]")
    print(f"component lower bound: {p.component_mass_lower_bound_kg():.3f} kg")
    print(f"arm length:            {p.arm_length_m:.4f} m   [from CAD]")
    print(f"max thrust/motor:      {p.max_thrust_per_motor_N:.2f} N   [EST - MEASURE THIS]")
    print(f"max total thrust:      {p.max_total_thrust_N:.2f} N")
    print(f"thrust-to-weight:      {p.thrust_to_weight:.2f}")
    print(f"flight controller:     {p.fc_firmware} on {p.fc_board}")
