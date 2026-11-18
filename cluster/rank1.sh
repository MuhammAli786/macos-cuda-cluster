#!/bin/bash
# Laptop side: start rank 1 in the background. Called over SSH by run.sh.
# usage: rank1.sh <model_dir> <split> [extra args]
# Expects ring_gen.py, gdn_cuda.py, moe_cpu.so and hosts.json in $HOME.
# PYTHON defaults to the venv this setup used.
PYTHON=${PYTHON:-/opt/omlx-cluster-worker/venv/bin/python}
pkill -f "$HOME/ring_gen.py" 2>/dev/null
sleep 1
MLX_RANK=1 MLX_HOSTFILE=$HOME/hosts.json nohup $PYTHON $HOME/ring_gen.py --model "$1" --split "$2" "${@:3}" \
    > $HOME/ring.log 2>&1 < /dev/null &
echo "rank1 pid $!"
