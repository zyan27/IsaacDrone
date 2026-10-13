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
    # [TODO(measure)] All-up weight: put the fully assembled drone WITH battery
    # on a kitchen scale. The 1.76 kg store figure is the SHIPPING/KIT weight,
    # not flight mass. A 4" quad like this is realistically ~0.5-0.7 kg AUW.
    mass_kg: float = 0.60  # [EST] placeholder until weighed

    # Component masses that are known, for a bottom-up sanity check vs the scale:
    motor_mass_kg: float = 0.028        # [SRC] DX2205, ~28 g each
    prop_mass_kg: float = 0.0035        # [SRC] Diatone 4040 3-blade, 3.5 g each
    battery_mass_kg: float = 0.178      # [SRC] 4S 1500 mAh pack, 178 g
    n_rotors: int = 4

    # --- Geometry --------------------------------------------------------
    # [TODO(measure)] Arm length = distance from the frame center to a motor
    # shaft. Measure motor-to-motor diagonal (wheelbase) and divide by 2.
    # 4040 props imply a ~180 mm wheelbase class, so ~0.09 m is the estimate.
    arm_length_m: float = 0.09          # [EST]

    # --- Propulsion ------------------------------------------------------
    # [TODO(measure)] Max thrust PER MOTOR at full throttle on a 4S pack.
    # Get it from a thrust stand or the motor/prop manufacturer chart.
    # DX2205 2300KV on 4S + 4040 is roughly ~1.0-1.2 kgf per motor.
    max_thrust_per_motor_N: float = 10.8  # [EST] ~1.1 kgf * 9.81
    kv: float = 2300.0                   # [SRC] motor KV
    battery_cells: int = 4               # [SRC] 4S
    battery_nominal_V: float = 14.8      # [SRC]

    # Yaw torque coefficient (reaction torque per unit thrust). [TODO] identify
    # from logs or a bench test; carried over from the reference repo for now.
    yaw_coeff: float = 1e-5              # [EST]

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


if __name__ == "__main__":
    p = DD24
    print(f"mass (AUW):            {p.mass_kg:.3f} kg   [EST - MEASURE THIS]")
    print(f"component lower bound: {p.component_mass_lower_bound_kg():.3f} kg")
    print(f"arm length:            {p.arm_length_m:.3f} m   [EST - MEASURE THIS]")
    print(f"max thrust/motor:      {p.max_thrust_per_motor_N:.2f} N   [EST - MEASURE THIS]")
    print(f"max total thrust:      {p.max_total_thrust_N:.2f} N")
    print(f"thrust-to-weight:      {p.thrust_to_weight:.2f}")
    print(f"flight controller:     {p.fc_firmware} on {p.fc_board}")
