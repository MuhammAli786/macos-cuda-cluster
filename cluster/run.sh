#!/bin/bash
# Start both ranks: the laptop (rank 1, first layers) in the background over
# SSH, then the Mac (rank 0, last layers) in the foreground.
#
# usage: MODEL=<dir> LAPTOP_ARGS="..." MAC_ARGS="..." ./run.sh <mac_layers,laptop_layers> [shared args...]
#   e.g. MODEL=Qwen3.6-35B-A3B-4bit LAPTOP_ARGS="--cpu-experts 6 --kv-bits 8" \
#        MAC_ARGS="--kv-bits 8" ./run.sh 19,21 --compile --prompt-file ../prompts/calib_prompt.txt
#
# Settings (environment):
#   LAPTOP      ssh target of the laptop           (default alimu@192.168.99.1)
#   MAC_PYTHON  python with MLX + MLX-LM on the Mac (default oMLX's bundled one)
#   MAC_MODELS  folder holding models on the Mac    (default ~/.omlx/models/mlx-community)
# The laptop expects ~/rank1.sh, ~/hosts.json, ~/moe_cpu.so and ~/models/<MODEL>.
O=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-Qwen3.6-35B-A3B-4bit}
LAPTOP=${LAPTOP:-alimu@192.168.99.1}
MAC_PYTHON=${MAC_PYTHON:-~/.omlx/bin/omlx-cluster-python}
MAC_MODELS=${MAC_MODELS:-~/.omlx/models/mlx-community}

pkill -f "ring_gen.py" 2>/dev/null
scp -q "$O/ring_gen.py" "$O/gdn_cuda.py" "$LAPTOP:"
ssh "$LAPTOP" "bash ~/rank1.sh ~/models/$MODEL $* $LAPTOP_ARGS" >/dev/null
cd "$O"
MLX_RANK=0 MLX_HOSTFILE="$O/hosts.json" $MAC_PYTHON ring_gen.py --model "$MAC_MODELS/$MODEL" \
    --split "$@" $MAC_ARGS > "$O/rank0.log" 2>&1
echo "=== Mac (rank 0)"; grep -vE "^\s*$" "$O/rank0.log" | tail -9
echo "=== laptop (rank 1)"; ssh "$LAPTOP" 'grep -E "^\[rank|Error|error" ~/ring.log | tail -8'
