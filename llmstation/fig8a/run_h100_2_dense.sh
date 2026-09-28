#!/bin/bash
set -euo pipefail

# Named for the eventual H100 x2 experiment; currently configured for Ada6000 x4.
# Set GPU_IDS=0,1 when moving to H100 x2; TP follows the visible GPU count.
# $1 - download dir
# $2 - request rate
# $3 - Number of tasklets performed in forward pass before preemption
# $4 - Wait in seconds after preemption in forward pass
# $5 - Number of tasklets performed in backward pass before preemption
# $6 - Wait in seconds after preemption in backward pass
# $7 - Optional model path (or set QWEN_MODEL)

if [[ $# -lt 6 ]]; then
    echo "Too few arguments."
    echo "Usage: $0 <download dir> <request rate> <forward tasklets> <forward wait> <backward tasklets> <backward wait> [qwen model]"
    exit 1
fi

download_dir=$1
request_rate=$2
forward_tasklets=$3
forward_wait=$4
backward_tasklets=$5
backward_wait=$6
qwen_model=${7:-${QWEN_MODEL:-/workspace/model/Qwen2.5-32B-Instruct}}
gpu_ids=${GPU_IDS:-0,1}
IFS=',' read -r -a gpu_devices <<< "$gpu_ids"
tp_size=${#gpu_devices[@]}
max_model_len=${MAX_MODEL_LEN:-8192}
# Leave GPU memory available for the LMS fine-tuning workers.
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.5}

benchmark_duration=240
num_prompts=$(awk \
    -v rate="$request_rate" \
    -v duration="$benchmark_duration" \
    'BEGIN {
        n = int(rate * duration + 0.5);
        if (n < 1) n = 1;
        print n;
    }')

vllm_server_log=fig8a_h100_2_dense_vllm_server.log
vllm_client_log=fig8a_h100_2_dense_vllm_client.log
lms_output_dir=$PWD
lms_log="${lms_output_dir}/lms.log"
vllm_pid=""

cleanup() {
    status=$?
    trap - EXIT
    if [[ -n "$vllm_pid" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
        kill "$vllm_pid" 2>/dev/null || true
        wait "$vllm_pid" 2>/dev/null || true
    fi
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT

rm -f "$vllm_server_log" "$vllm_client_log" "$lms_log"

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export CUDA_MPS_PIPE_DIRECTORY=./nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=./nvidia-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

nohup vllm serve "$qwen_model" \
    -tp="$tp_size" --download_dir "$download_dir" \
    --disable-async-output-proc --disable-log-requests \
    --max-model-len="$max_model_len" \
    --gpu-memory-utilization="$gpu_memory_utilization" \
    --enable-lora --max-loras 4 --max-lora-rank=8 \
    --enable-lms --lms-output="$lms_output_dir" \
    --lms-forward-tasklets "$forward_tasklets" \
    --lms-forward-wait "$forward_wait" \
    --lms-backward-tasklets "$backward_tasklets" \
    --lms-backward-wait "$backward_wait" \
    > "$vllm_server_log" 2>&1 &
vllm_pid=$!
echo "vLLM runs in process ${vllm_pid} on GPUs ${gpu_ids} (TP=${tp_size})."

# INIT_SECS is a timeout, not a fixed delay. Multi-GPU model sharing and
# CUDA graph capture can take several minutes before LMS starts training.
init_secs=${INIT_SECS:-1200}
init_started=$SECONDS
init_deadline=$((init_started + init_secs))
next_status=$init_started
echo "Wait up to $init_secs seconds for the API and LMS training to be ready."

while true; do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
        echo "vLLM exited during initialization; check $vllm_server_log."
        tail -n 60 "$vllm_server_log" >&2 || true
        exit 1
    fi

    api_ready=no
    if python -c '
import sys
import urllib.request

try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8000/health", timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
' >/dev/null 2>&1; then
        api_ready=yes
    fi
    lms_ready=no
    if [[ -s "$lms_log" ]]; then
        lms_ready=yes
    fi

    if [[ "$api_ready" == yes && "$lms_ready" == yes ]]; then
        echo "API and LMS training are ready after $((SECONDS - init_started)) seconds."
        break
    fi

    if (( SECONDS >= init_deadline )); then
        echo "Initialization timed out after ${init_secs}s: API=${api_ready}, LMS log=${lms_ready}."
        echo "The vLLM process is still alive; cleanup will stop it. Check $vllm_server_log."
        tail -n 60 "$vllm_server_log" >&2 || true
        exit 1
    fi

    if (( SECONDS >= next_status )); then
        echo "Still initializing ($((SECONDS - init_started))/${init_secs}s): API=${api_ready}, LMS log=${lms_ready}."
        next_status=$((SECONDS + 30))
    fi
    sleep 5
done

start_line=$(wc -l < "$lms_log")
burstgpt_trace="$download_dir/BurstGPT_half.csv"

python ../../python/benchmark_serving.py \
    --backend vllm --model "$qwen_model" \
    --dataset-name burstgpt --dataset-path "$burstgpt_trace" \
    --burstgpt-max-model-len "$max_model_len" --ignore-eos \
    --request-rate "$request_rate" --num-prompts "$num_prompts" \
    > "$vllm_client_log"

end_line=$(wc -l < "$lms_log")

python ../parse_log.py \
    --vllm-log "$vllm_client_log" --lms-log "$lms_log" \
    --lms-forward-tasklets "$forward_tasklets" \
    --lms-forward-wait "$forward_wait" \
    --lms-backward-tasklets "$backward_tasklets" \
    --lms-backward-wait "$backward_wait" \
    --start-line "$start_line" --end-line "$end_line"
