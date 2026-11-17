#!/usr/bin/env bash
# Convert assets/dd24/dd24.urdf -> assets/dd24/dd24.usd with Isaac Lab's importer.
#
# Local Linux box:   ISAACLAB=~/IsaacLab bash assets/dd24/convert_to_usd.sh
# Oscar:             sbatch oscar/convert_dd24.sbatch   (wraps this in the container)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB="${ISAACLAB:-${ISAACLAB_DIR:-$HOME/IsaacLab}}"
URDF="${1:-$HERE/dd24.urdf}"
USD="${2:-$HERE/dd24.usd}"

[ -f "$URDF" ] || { echo "No URDF at $URDF - run build_dd24_urdf.py first"; exit 1; }

# Props: no joint drive (free-spinning visuals). Base: floating.
"$ISAACLAB/isaaclab.sh" -p "$ISAACLAB/scripts/tools/convert_urdf.py" \
    "$URDF" "$USD" \
    --joint-target-type none \
    --joint-stiffness 0.0 \
    --joint-damping 0.0 \
    --headless

echo "wrote $USD"
