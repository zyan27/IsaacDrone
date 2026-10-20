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
    mass_kg: float = 0.82704  # [SRC] user-measured AUW (827.04 g)

    # Component masses that are known, for a bottom-up sanity check vs the scale:
    motor_mass_kg: float = 0.028        # [SRC] DX2205, ~28 g each
    prop_mass_kg: float = 0.0035        # [SRC] Diatone 4040 3-blade, 3.5 g each
    battery_mass_kg: float = 0.178      # [SRC] 4S 1500 mAh pack, 178 g
    n_rotors: int = 4

    # --- Geometry --------------------------------------------------------
    # Arm length = frame center to a motor shaft. User-measured motor-to-motor
    # diagonal 0.23 m, so 0.115 m. Square X frame: each motor sits 0.115/sqrt(2)
    # = 0.0813 m forward/back and left/right of center (see lever_arm_m).
    arm_length_m: float = 0.115         # [SRC] user-measured diagonal / 2
    # Diatone 4040 3-blade: 4.0" diameter, 4.0" pitch. Adjacent motors are
    # 0.163 m apart, so ~6 cm tip-to-tip clearance.
    prop_diameter_m: float = 0.1016     # [SRC]

    # --- Propulsion ------------------------------------------------------
    # [TODO(measure)] Max thrust PER MOTOR at full throttle on a 4S pack.
    # Get it from a thrust stand or the motor/prop manufacturer chart.
    # Community bench data for 2205 2300KV + 4"/4.5" triblades on 4S: ~700-800 gf
    # per motor at full throttle on a fresh pack (TWR ~3.6 at 827 g).
    max_thrust_per_motor_N: float = 7.36  # [EST] 0.75 kgf * 9.81, bench data, not this drone
    kv: float = 2300.0                   # [SRC] motor KV
    battery_cells: int = 4               # [SRC] 4S
    battery_nominal_V: float = 14.8      # [SRC]

    # Inertia about the CG, body axes. [TODO(measure)] bifilar pendulum. Expected
    # range for an 827 g 4" quad: Ixx~Iyy 2-4e-3, Izz 4-6e-3. PX4's default rate
    # gains are only stable here for Ixx >~ 3e-3 at this thrust (see
    # sim_px4/check_rate_stability.py), and the real DD24 flies on those gains.
    # The real DD24 hovers steadily on those defaults (user, 2026-10-20), so
    # the estimate sits in the stable region rather than the bottom-up ~1.7e-3.
    inertia_kgm2: tuple = (5.0e-3, 5.0e-3, 8.0e-3)  # [EST]
    # First-order motor/ESC thrust lag. Typical 2205 + BLHeli: 15-30 ms.
    motor_time_constant_s: float = 0.02  # [EST]

    # Thrust vs ESC command: thrust ~ Fmax * u**2 (prop thrust ~ rpm^2, rpm ~ u).
    # Predicts hover at u ~0.52; user flight: 0.40 barely lifts, 0.70 climbs.
    thrust_curve_exponent: float = 2.0   # [EST]
    # Prop reaction torque per newton of thrust (m). Typical 4" props: 0.01-0.02.
    # The reference repo's 1e-5 gave the DD24 almost no yaw authority.
    yaw_coeff: float = 0.016             # [EST]

    # --- Flight controller ----------------------------------------------
    fc_board: str = "mamba-f405-mk2"     # [SRC] PX4 target
    fc_firmware: str = "PX4"             # [SRC] DD24-B runs PX4

    # --- Derived helpers -------------------------------------------------
    @property
    def weight_N(self) -> float:
        return self.mass_kg * 9.81

    @property
    def lever_arm_m(self) -> float:
        """Roll/pitch moment arm of each motor on a square X frame."""
        return self.arm_length_m / 2 ** 0.5

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


if __name__ == "__main__":
    p = DD24
    print(f"mass (AUW):            {p.mass_kg:.3f} kg   [user-measured]")
    print(f"component lower bound: {p.component_mass_lower_bound_kg():.3f} kg")
    print(f"arm length:            {p.arm_length_m:.3f} m   (lever {p.lever_arm_m:.4f} m)")
    print(f"max thrust/motor:      {p.max_thrust_per_motor_N:.2f} N   [EST - MEASURE THIS]")
    print(f"max total thrust:      {p.max_total_thrust_N:.2f} N")
    print(f"thrust-to-weight:      {p.thrust_to_weight:.2f}")
    print(f"flight controller:     {p.fc_firmware} on {p.fc_board}")
