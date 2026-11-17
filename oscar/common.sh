#!/usr/bin/env bash
# Shared Oscar settings + a wrapper that runs a command inside the Isaac Sim
# container. Sourced by the *.sbatch files; override any variable via `export`.

ISAAC_SIM_TAG="${ISAAC_SIM_TAG:-5.0.0}"
# Isaac Lab v2.2.x is the release line built for Isaac Sim 5.0. Do NOT use `main`:
# it has moved on to newer Isaac Sim versions and the imports will break.
ISAACLAB_TAG="${ISAACLAB_TAG:-v2.2.1}"

# Resolve ~/scratch to its real path (/oscar/scratch/$USER) so it can be bound.
SCRATCH_ROOT="$(readlink -f "${SCRATCH_ROOT:-$HOME/scratch}")"
ISAAC_ROOT="${ISAAC_ROOT:-$SCRATCH_ROOT/isaac}"
SIF="${SIF:-$ISAAC_ROOT/isaac-sim-${ISAAC_SIM_TAG}.sif}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$ISAAC_ROOT/IsaacLab}"
REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
ISAAC_CACHE="${ISAAC_CACHE:-$ISAAC_ROOT/cache}"
# The .sif is read-only, so Isaac Lab + its pip deps are installed as a Python
# "user" install here (on scratch, not your small home quota).
PYUSER="${PYUSER:-$ISAAC_ROOT/pyuser}"

isaac_exec() {
    mkdir -p "$ISAAC_CACHE"/{kit,kitdata,kitlogs,ov,pip,glcache,computecache,logs,data,documents} "$PYUSER"
    # Apptainer runs as YOU (not root), so Omniverse caches land under $HOME
    # inside the container. Redirect each of them to scratch.
    # ISAAC_TIMEOUT (e.g. 30m) kills the container if it overruns; unset = no limit.
    timeout --kill-after=60 "${ISAAC_TIMEOUT:-0}" \
    apptainer exec --nv \
        --env ACCEPT_EULA=Y \
        --env OMNI_KIT_ACCEPT_EULA=YES \
        --env PRIVACY_CONSENT=Y \
        --env PYTHONUSERBASE="$PYUSER" \
        --env PATH="$PYUSER/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        --bind "$SCRATCH_ROOT" \
        --bind "$REPO_DIR" \
        --bind "$ISAAC_CACHE/kit:/isaac-sim/kit/cache" \
        --bind "$ISAAC_CACHE/kitdata:/isaac-sim/kit/data" \
        --bind "$ISAAC_CACHE/kitlogs:/isaac-sim/kit/logs" \
        --bind "$ISAAC_CACHE/ov:$HOME/.cache/ov" \
        --bind "$ISAAC_CACHE/pip:$HOME/.cache/pip" \
        --bind "$ISAAC_CACHE/glcache:$HOME/.cache/nvidia/GLCache" \
        --bind "$ISAAC_CACHE/computecache:$HOME/.nv/ComputeCache" \
        --bind "$ISAAC_CACHE/logs:$HOME/.nvidia-omniverse/logs" \
        --bind "$ISAAC_CACHE/data:$HOME/.local/share/ov/data" \
        --bind "$ISAAC_CACHE/documents:$HOME/Documents" \
        "$SIF" \
        env -u BASH_ENV -u 'BASH_FUNC_module%%' -u 'BASH_FUNC_ml%%' \
        bash -c "$*"
    # (env -u ...: Oscar exports its Lmod `module` function into every shell; it
    #  points at host paths that don't exist in the container, so drop it.)
}
