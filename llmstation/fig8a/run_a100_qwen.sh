#!/bin/bash
set -euo pipefail

# $1 - download dir
# $2 - request rate
# $3 - Number of tasklets performed in forward pass before preemption
# $4 - Wait in seconds after preemption in forward pass
# $5 - Number of tasklets performed in backward pass before preemption
# $6 - Wait in seconds after preemption in backward pass
# $7 - Optional Qwen model path (or set QWEN_MODEL)

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
qwen_model=${7:-${QWEN_MODEL:-/root/autodl-tmp/Qwen2.5-14B-Instruct}}
gpu_id=${GPU_ID:-0}
max_model_len=${MAX_MODEL_LEN:-8192}

benchmark_duration=240
num_prompts=$(awk \
    -v rate="$request_rate" \
    -v duration="$benchmark_duration" \
    'BEGIN {
        n = int(rate * duration + 0.5);
        if (n < 1) n = 1;
        print n;
    }')

vllm_server_log=fig8a_a100_qwen_vllm_server.log
vllm_client_log=fig8a_a100_qwen_vllm_client.log
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

export CUDA_VISIBLE_DEVICES="$gpu_id"
export CUDA_MPS_PIPE_DIRECTORY=./nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=./nvidia-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

nohup vllm serve "$qwen_model" \
    -tp=1 --download_dir "$download_dir" \
    --disable-async-output-proc --disable-log-requests \
    --max-model-len="$max_model_len" --gpu-memory-utilization=0.75 \
    --enable-lora --max-loras 4 --max-lora-rank=8 \
    --enable-lms --lms-output="$lms_output_dir" \
    --lms-forward-tasklets "$forward_tasklets" \
    --lms-forward-wait "$forward_wait" \
    --lms-backward-tasklets "$backward_tasklets" \
    --lms-backward-wait "$backward_wait" \
    > "$vllm_server_log" 2>&1 &
vllm_pid=$!
echo "vLLM runs in process ${vllm_pid} on GPU ${gpu_id}."

init_secs=${INIT_SECS:-120}
echo "Wait $init_secs seconds for vLLM initialization before benchmarking."
sleep "$init_secs"

if ! kill -0 "$vllm_pid" 2>/dev/null || [[ ! -f "$lms_log" ]]; then
    echo "LLMStation failed to initialize; check $vllm_server_log."
    exit 1
fi

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

