#!/usr/bin/env bash
set -e

cd /root/autodl-tmp/llm_reasoning_alignment

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export HF_HUB_DISABLE_XET=1

LOG_DIR=outputs/logs/grpo_v9_best
mkdir -p "$LOG_DIR"

MODEL_DIR=$(find /root/autodl-tmp/hf_cache /root/.cache/huggingface -path "*models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/config.json" 2>/dev/null | head -n 1 | xargs dirname)

if [ -z "$MODEL_DIR" ]; then
  MODEL_REF="Qwen/Qwen2.5-1.5B-Instruct"
else
  MODEL_REF="$MODEL_DIR"
fi

echo "MODEL_REF=$MODEL_REF"

test -d outputs/checkpoints/sft_lora_v5_eval_style_completion_only
test -f data/processed/grpo_v5_mixed_replay_from_sft_v5.jsonl
test -f scripts/87_train_grpo_v5_from_sft_v5_server.py
test -f scripts/01_lmeval_eval.py

cat > configs/grpo_v9_quick75_stronger_conservative.yaml <<YAML
base_model: $MODEL_REF

start_adapter_path: outputs/checkpoints/sft_lora_v5_eval_style_completion_only
train_file: data/processed/grpo_v5_mixed_replay_from_sft_v5.jsonl
output_dir: outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative

max_prompt_length: 320
max_completion_length: 384

max_steps: 75

per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.0000008

num_generations: 4
temperature: 0.7
beta: 0.7

logging_steps: 1
save_steps: 25
save_total_limit: 4

dtype: bf16
YAML

echo "====== 训练 GRPO_v9 quick75 stronger conservative ======"
if [ -f outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative/adapter_config.json ]; then
  echo "GRPO_v9 final adapter 已存在，跳过训练。"
else
  rm -rf outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative
  python scripts/87_train_grpo_v5_from_sft_v5_server.py --config configs/grpo_v9_quick75_stronger_conservative.yaml 2>&1 | tee "$LOG_DIR/train.log"
fi

echo "====== 收集所有候选 checkpoint ======"
CANDIDATES=()

if [ -f outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative/adapter_config.json ]; then
  CANDIDATES+=("final:outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative")
fi

for ckpt in outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative/checkpoint-*; do
  if [ -f "$ckpt/adapter_config.json" ]; then
    name=$(basename "$ckpt")
    CANDIDATES+=("${name}:${ckpt}")
  fi
done

echo "候选 checkpoint:"
printf '%s\n' "${CANDIDATES[@]}"

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "没有找到任何可评估 checkpoint。"
  exit 1
fi

echo "====== 开始评估所有 checkpoint：limit=100 和 limit=200 ======"

SUMMARY="$LOG_DIR/summary_paths.txt"
rm -f "$SUMMARY"

for item in "${CANDIDATES[@]}"; do
  NAME="${item%%:*}"
  PEFT_PATH="${item#*:}"

  SAFE_NAME=$(echo "$NAME" | sed 's/[^A-Za-z0-9_-]/_/g')

  echo "====== eval $NAME limit=100 ======"
  cat > "configs/eval_grpo_v9_${SAFE_NAME}_limit100.yaml" <<YAML
base_model: $MODEL_REF
peft_path: $PEFT_PATH

tasks: gsm8k_cot
limit: 100
batch_size: 1
device: cuda
dtype: bfloat16
apply_chat_template: true

output_name: grpo_v9_${SAFE_NAME}_qwen25_15b_gsm8k_cot_limit100
YAML

  python scripts/01_lmeval_eval.py --config "configs/eval_grpo_v9_${SAFE_NAME}_limit100.yaml" 2>&1 | tee "$LOG_DIR/eval_${SAFE_NAME}_limit100.log"

  echo "limit100 $NAME outputs/eval/grpo_v9_${SAFE_NAME}_qwen25_15b_gsm8k_cot_limit100" >> "$SUMMARY"

  echo "====== eval $NAME limit=200 ======"
  cat > "configs/eval_grpo_v9_${SAFE_NAME}_limit200.yaml" <<YAML
base_model: $MODEL_REF
peft_path: $PEFT_PATH

tasks: gsm8k_cot
limit: 200
batch_size: 1
device: cuda
dtype: bfloat16
apply_chat_template: true

output_name: grpo_v9_${SAFE_NAME}_qwen25_15b_gsm8k_cot_limit200
YAML

  python scripts/01_lmeval_eval.py --config "configs/eval_grpo_v9_${SAFE_NAME}_limit200.yaml" 2>&1 | tee "$LOG_DIR/eval_${SAFE_NAME}_limit200.log"

  echo "limit200 $NAME outputs/eval/grpo_v9_${SAFE_NAME}_qwen25_15b_gsm8k_cot_limit200" >> "$SUMMARY"
done

echo "====== 全部评估完成 ======"
echo "日志目录: $LOG_DIR"
echo "结果路径汇总: $SUMMARY"
echo ""
echo "快速查看分数："
grep -R "flexible-extract" "$LOG_DIR"/eval_*_limit*.log || true
grep -R "strict-match" "$LOG_DIR"/eval_*_limit*.log || true
