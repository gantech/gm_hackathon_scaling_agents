#!/bin/bash
# Python for GINO training on Polaris: ALCF's conda base (optimized torch) plus the aifiv_env
# venv layered on it with torch_geometric. The agents call this as --train-python, so only the
# training subprocesses load conda; the agents keep their own uv .venv.
#
#   scaling_agents/polaris_train_python.sh train.py --device cuda ...
#
# AIFIV_ENV overrides the venv (default: aifiv/aifiv_env in this repo). The venv was made with
#   module use /soft/modulefiles; module load conda; conda activate base
#   python -m venv --system-site-packages aifiv/aifiv_env
#   source aifiv/aifiv_env/bin/activate; pip install torch_geometric

set -eo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AIFIV_ENV=${AIFIV_ENV:-$REPO/aifiv/aifiv_env}

# Leave the agents' uv venv behind before loading conda.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
type module > /dev/null 2>&1 || source /etc/profile
module use /soft/modulefiles
module load conda
conda activate base
source "$AIFIV_ENV/bin/activate"

exec python "$@"
