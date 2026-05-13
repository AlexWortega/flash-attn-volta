#!/usr/bin/env bash
# Run the full verify suite and gather output into results/.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

cd "$(dirname "$0")/.."
mkdir -p results

echo "=== correctness ==="     | tee  results/verify.log
python3 tests/test_correctness.py 2>&1 | tee -a results/verify.log
echo                            | tee -a results/verify.log
echo "=== stability ==="        | tee -a results/verify.log
python3 bench/stability.py      2>&1 | tee -a results/verify.log
echo                            | tee -a results/verify.log
echo "=== memory ==="           | tee -a results/verify.log
python3 bench/memory.py         2>&1 | tee -a results/verify.log
echo                            | tee -a results/verify.log
echo "=== benchmark ==="        | tee -a results/verify.log
python3 bench/bench.py          2>&1 | tee -a results/verify.log
echo "=== gpu cap ==="          | tee -a results/verify.log
python3 -c "import torch; cap=torch.cuda.get_device_capability(0); print('compute capability', cap); print('device name', torch.cuda.get_device_name(0)); assert cap == (7, 0), 'NOT SM 7.0'" 2>&1 | tee -a results/verify.log
