#!/usr/bin/env bash
# One-shot MetaWorld setup for USC CARC Discovery. Run on a LOGIN node (needs internet):
#   bash carc/setup_carc.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
mkdir -p "$FSPREF_ROOT/data"

if [ ! -d "$FSPREF_ROOT/env" ]; then
    # MetaWorld 3 supports 3.9-3.12; 3.11 matched the ManiSkill3 setup that worked here.
    conda create -y -p "$FSPREF_ROOT/env" -c conda-forge python=3.11
fi
# osmesa gives CPU-only offscreen rendering for MuJoCo on nodes without EGL.
# Optional: only needed for --render jobs, never for simulation.
# `osmesa` carries libOSMesa.so; modern `mesalib` alone does not.
conda install -y -p "$FSPREF_ROOT/env" -c conda-forge mesalib osmesa libglu 2>/dev/null || true

fspref_activate
pip install --upgrade pip
pip install torch                      # default CUDA 12 wheel; cluster drivers are 12.x
pip install -r ../requirements.txt

python - <<'PY'
import metaworld, mujoco, gymnasium, numpy, torch
print("metaworld", getattr(metaworld, "__version__", "?"))
print("mujoco", mujoco.__version__, "| gymnasium", gymnasium.__version__,
      "| numpy", numpy.__version__, "| torch", torch.__version__, "cuda", torch.cuda.is_available())
import metaworld.policies as P
print("window policies:", [n for n in dir(P) if "Window" in n])
PY
echo
echo "Setup complete. FSPREF_ROOT=$FSPREF_ROOT"
echo "activate: source carc/env.sh && fspref_activate"
