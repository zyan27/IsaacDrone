"""Decimate the DD24 CAD STL, convert mm->m, rotate so the long axis is +x (forward), and
move the origin to the estimated CG. Writes a light OBJ for the URDF/USD."""
import sys, numpy as np, trimesh
# usage: python assets/dd24/prep_cad_mesh.py DD24.stl assets/dd24/meshes/dd24_body.obj 60000
# needs: pip install trimesh
src, dst, target_faces = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = trimesh.load(src, force="mesh", process=False)
m.merge_vertices()
# Vertex clustering: snap vertices to a grid, merge, drop collapsed faces. Robust on CAD
# assemblies with hundreds of disjoint bodies (quadric decimation stalls on those).
def cluster(vin, fin, cell):
    key = np.floor(vin / cell).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    vout = np.zeros((len(uniq), 3)); np.add.at(vout, inv, vin)
    vout /= np.bincount(inv)[:, None]
    fout = inv[fin]
    ok = (fout[:, 0] != fout[:, 1]) & (fout[:, 1] != fout[:, 2]) & (fout[:, 0] != fout[:, 2])
    fout = fout[ok]
    return vout, fout
cell = 0.5
while True:
    v, f = cluster(m.vertices, m.faces, cell)
    if len(f) <= target_faces: break
    cell *= 1.25
print(f"grid cell {cell:.2f} mm")
v = v / 1000.0                                            # mm -> m
# The camera (with its printed mount and ribbon cable) sits at the CAD -y tip, and the
# camera side is the front. Rotate +90 deg about z: CAD (x, y) -> (-y, x), so CAD -y
# becomes +x (forward), matching the body frame used by the env (x fwd, y left, z up).
# This also lines the long axis up with roll (x), consistent with measured Iyy > Ixx.
v = np.stack([-v[:, 1], v[:, 0], v[:, 2]], axis=1)
cg = np.array([0.0, 0.0, 0.0537])                         # bbox centre; [EST] CG height
v -= cg
out = trimesh.Trimesh(v, f, process=True)
out.export(dst)
print("faces", len(out.faces), "bounds (m)", out.bounds.round(4).tolist(), "-> wrote", dst)
