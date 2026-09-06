# Shared environment for the few-shot preference RL project on USC CARC Discovery.
# Source this, don't execute it:  source carc/env.sh && fspref_activate
#
# FSPREF_ROOT holds the conda env and data. Default is scratch (purged periodically);
# override to a project dir for durable data, e.g.
#   FSPREF_ROOT=/project2/biyik_1165/$USER/fspref source carc/env.sh

export FSPREF_ROOT="${FSPREF_ROOT:-/scratch1/$USER/fspref}"

# Non-interactive shells (ssh discovery '<cmd>') don't initialize Lmod.
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module purge
module load conda

fspref_activate() {
    eval "$(conda shell.bash hook)"
    conda activate "$FSPREF_ROOT/env"
    # Login nodes: OpenBLAS tries to spawn 64 threads and hits the per-user
    # process limit; pin it. Compute jobs can raise this.
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    # PyOpenGL/mujoco dlopen libOSMesa/libEGL from the conda env, which is not
    # on the default search path.
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # NOTE: MUJOCO_GL is deliberately NOT set here. mujoco imports its GL
    # backend eagerly when MUJOCO_GL is defined, and login/CPU nodes have no
    # working OSMesa or EGL, so setting it breaks plain `import mujoco`.
    # Simulation never needs a GL context; call fspref_render_setup only for
    # jobs that actually render frames.
    unset MUJOCO_GL PYOPENGL_PLATFORM
}

fspref_render_setup() {
    # Opt-in headless rendering: EGL when a GPU is allocated, osmesa otherwise.
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export MUJOCO_GL=egl
    else
        export MUJOCO_GL=osmesa
    fi
    export PYOPENGL_PLATFORM="$MUJOCO_GL"
}
