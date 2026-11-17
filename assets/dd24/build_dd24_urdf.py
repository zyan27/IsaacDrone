"""Build a DD24 URDF from your CAD mesh + measured mass/inertia.

Pure Python (numpy only) - run it anywhere, including the Mac. Then convert the
URDF to USD with convert_to_usd.sh (needs Isaac Lab, i.e. Oscar or a Linux box).

Link/joint names match what DroneHoverEnv expects:
    body                       <- the airframe, carries (mass - props) and inertia
    m1_prop .. m4_prop         <- thin discs at the rotor positions
    m1_joint .. m4_joint       <- continuous joints body -> prop

Motor numbering matches the mixer in drone_hover_env.py (x forward, y left):
    m1 (+x, -y) front-right   m2 (-x, -y) back-right
    m3 (-x, +y) back-left     m4 (+x, +y) front-left

The body frame is placed at the CENTRE OF GRAVITY. If your CAD origin is not the
CG, pass --cg (in mesh coordinates, metres after --mesh-scale) and the mesh is
shifted so the CG lands on the body origin.

Examples
--------
# placeholder box (no mesh yet), values from environments/dd24_params.py
python assets/dd24/build_dd24_urdf.py

# with CAD mesh exported in millimetres, CG measured in CAD
python assets/dd24/build_dd24_urdf.py \
    --mesh /path/to/dd24_frame.stl --mesh-scale 0.001 \
    --cg 0.0 0.0 0.012 \
    --inertia 3.9e-3 4.1e-3 7.2e-3     # Ixx Iyy Izz [Ixy Ixz Iyz], kg*m^2 about CG
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from environments.dd24_params import DD24  # noqa: E402

PROP_RADIUS = 0.0508      # 4" prop = 0.1016 m diameter
PROP_THICKNESS = 0.005


def prop_positions(center_to_motor: float, z: float) -> np.ndarray:
    s = center_to_motor / np.sqrt(2.0)
    return np.array([
        [+s, -s, z],   # m1 front-right
        [-s, -s, z],   # m2 back-right
        [-s, +s, z],   # m3 back-left
        [+s, +s, z],   # m4 front-left
    ])


def disc_inertia(m: float, r: float, h: float) -> np.ndarray:
    ixx = m * (3 * r * r + h * h) / 12.0
    return np.diag([ixx, ixx, 0.5 * m * r * r])


def body_inertia_minus_props(I_total, m_prop, positions, I_prop_own):
    """The measured inertia is for the whole drone about its CG. Props are separate
    links, so remove their contribution (parallel-axis) from the body link."""
    I_body = I_total.copy()
    for r in positions:
        I_body -= m_prop * (np.dot(r, r) * np.eye(3) - np.outer(r, r)) + I_prop_own
    return I_body


def fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def build_urdf(args) -> str:
    mass_total = args.mass
    m_prop = args.prop_mass
    if args.inertia is None:
        I = np.array([[DD24.Ixx, DD24.Ixy, DD24.Ixz],
                      [DD24.Ixy, DD24.Iyy, DD24.Iyz],
                      [DD24.Ixz, DD24.Iyz, DD24.Izz]])
    else:
        v = list(args.inertia) + [0.0] * (6 - len(args.inertia))
        ixx, iyy, izz, ixy, ixz, iyz = v
        I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    pos = prop_positions(args.arm_length, args.prop_z)
    I_prop = disc_inertia(m_prop, PROP_RADIUS, PROP_THICKNESS)
    I_body = body_inertia_minus_props(I, m_prop, pos, I_prop)
    m_body = mass_total - 4 * m_prop

    eig = np.linalg.eigvalsh(I_body)
    if m_body <= 0 or np.any(eig <= 0):
        raise SystemExit(f"Body inertia not positive-definite after removing props "
                         f"(eig={eig}); check units (kg*m^2) and prop mass.")
    # Triangle inequality for principal moments (physically realisable body).
    a, b, c = eig
    if not (a + b >= c - 1e-12):
        print(f"WARNING: principal moments {eig} violate the triangle inequality - "
              f"inertia values are probably wrong.")

    # Visual / collision geometry for the body
    cg = np.array(args.cg)
    if args.mesh:
        mesh_name = os.path.basename(args.mesh)
        dst = os.path.join(HERE, "meshes", mesh_name)
        if os.path.abspath(args.mesh) != os.path.abspath(dst):
            shutil.copy(args.mesh, dst)
        sc = args.mesh_scale
        visual_geom = f'<mesh filename="meshes/{mesh_name}" scale="{sc} {sc} {sc}"/>'
        visual_origin = f'<origin xyz="{fmt(-cg)}" rpy="{fmt(args.mesh_rpy)}"/>'
    else:
        visual_geom = f'<box size="{fmt(args.collision_box)}"/>'
        visual_origin = '<origin xyz="0 0 0" rpy="0 0 0"/>'
    # Collision is always a simple box: fast, stable, and good enough for ground
    # contact (the net is a geofence in the env, not a collider).
    collision_geom = f'<box size="{fmt(args.collision_box)}"/>'

    def inertia_tag(Im):
        return (f'<inertia ixx="{Im[0,0]:.9g}" ixy="{Im[0,1]:.9g}" ixz="{Im[0,2]:.9g}" '
                f'iyy="{Im[1,1]:.9g}" iyz="{Im[1,2]:.9g}" izz="{Im[2,2]:.9g}"/>')

    props = []
    for i, p in enumerate(pos, start=1):
        props.append(f"""
  <link name="m{i}_prop">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{m_prop:.9g}"/>
      {inertia_tag(I_prop)}
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="{PROP_RADIUS}" length="{PROP_THICKNESS}"/></geometry>
    </visual>
  </link>
  <joint name="m{i}_joint" type="continuous">
    <parent link="body"/>
    <child link="m{i}_prop"/>
    <origin xyz="{fmt(p)}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>""")

    urdf = f"""<?xml version="1.0"?>
<!-- Generated by assets/dd24/build_dd24_urdf.py - do not hand-edit, re-run the script.
     total mass {mass_total:.5f} kg | arm {args.arm_length:.4f} m
     whole-drone inertia about CG (kg m^2): Ixx={I[0,0]:.4e} Iyy={I[1,1]:.4e} Izz={I[2,2]:.4e} -->
<robot name="dd24">
  <link name="body">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{m_body:.9g}"/>
      {inertia_tag(I_body)}
    </inertial>
    <visual>
      {visual_origin}
      <geometry>{visual_geom}</geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>{collision_geom}</geometry>
    </collision>
  </link>{''.join(props)}
</robot>
"""
    print(f"total mass {mass_total:.4f} kg  (body {m_body:.4f} + 4 x prop {m_prop:.4f})")
    print(f"whole-drone I diag {np.diag(I)}  -> body link I diag {np.diag(I_body)}")
    return urdf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", help="Body mesh (.stl/.obj/.dae). STEP must be converted to STL first.")
    ap.add_argument("--mesh-scale", type=float, default=1.0, help="0.001 if the CAD export is in mm")
    ap.add_argument("--mesh-rpy", type=float, nargs=3, default=[0, 0, 0],
                    help="Rotate mesh into x-forward / z-up (radians)")
    ap.add_argument("--cg", type=float, nargs=3, default=[0, 0, 0],
                    help="CG location in mesh coordinates (m, after scale)")
    ap.add_argument("--mass", type=float, default=DD24.mass_kg)
    ap.add_argument("--inertia", type=float, nargs="+",
                    help="Ixx Iyy Izz [Ixy Ixz Iyz] about the CG, kg*m^2 (default: dd24_params)")
    ap.add_argument("--arm-length", type=float, default=DD24.arm_length_m, help="centre-to-motor (m)")
    ap.add_argument("--prop-mass", type=float, default=DD24.prop_mass_kg)
    ap.add_argument("--prop-z", type=float, default=0.02, help="prop plane height above CG (m)")
    ap.add_argument("--collision-box", type=float, nargs=3, default=[0.18, 0.18, 0.06])
    ap.add_argument("--out", default=os.path.join(HERE, "dd24.urdf"))
    args = ap.parse_args()
    if args.inertia is not None and len(args.inertia) not in (3, 6):
        ap.error("--inertia takes 3 or 6 values")

    urdf = build_urdf(args)
    with open(args.out, "w") as f:
        f.write(urdf)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
