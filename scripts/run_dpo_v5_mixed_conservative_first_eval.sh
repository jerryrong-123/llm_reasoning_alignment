#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm_reasoning_alignment

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export HF_HUB_DISABLE_XET=1

MAIN_LOG_DIR=outputs/logs/dpo_v5_mixed_conservative_first_eval
mkdir -p "$MAIN_LOG_DIR"

MODEL_CONFIG=$(find /root/autodl-tmp/hf_cache /root/.cache/huggingface -path "*models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/config.json" 2>/dev/null | head -n 1)

if [ -z "$MODEL_CONFIG" ]; then
  MODEL_REF="Qwen/Qwen2.5-1.5B-Instruct"
else
  MODEL_REF=$(dirname "$MODEL_CONFIG")
fi

echo "MODEL_REF=$MODEL_REF"

test -d outputs/checkpoints/sft_lora_v5_eval_style_completion_only
test -f data/processed/dpo_v5_mixed_conservative.jsonl
test -f scripts/89_train_dpo_v4_targeted_server.py
test -f scripts/01_lmeval_eval.py

NAME="dpo_v5_mixed_conservative_from_sft_v5"
OUT_DIR="outputs/checkpoints/${NAME}"

rm -rf "$OUT_DIR"

cat > "configs/${NAME}.yaml" <<YAML
base_model: $MODEL_REF

sft_adapter_path: outputs/checkpoints/sft_lora_v5_eval_style_completion_only
train_file: data/processed/dpo_v5_mixed_conservative.jsonl
output_dir: $OUT_DIR

max_steps: 50
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 0.0000002
beta: 0.1

max_length: 768
max_prompt_length: 320
max_completion_length: 448

logging_steps: 1
save_steps: 50
save_total_limit: 2

dtype: bf16
warmup_ratio: 0.03
lr_scheduler_type: linear
gradient_checkpointing: false
YAML

echo "============================================================"
echo "开始训练 DPO_v5 mixed conservative"
echo "============================================================"

python scripts/89_train_dpo_v4_targeted_server.py --config "configs/${NAME}.yaml" 2>&1 | tee "$MAIN_LOG_DIR/train.log"

echo "============================================================"
echo "训练完成，开始评估 checkpoint-50 limit=100"
echo "============================================================"

CKPT_NAME="checkpoint-50"
PEFT_PATH="$OUT_DIR/checkpoint-50"

if [ ! -f "$PEFT_PATH/adapter_config.json" ]; then
  echo "找不到 checkpoint-50 adapter_config.json"
  exit 1
fi

CONFIG_PATH="configs/eval_${NAME}_${CKPT_NAME}_limit100.yaml"
LOG_PATH="$MAIN_LOG_DIR/eval_${CKPT_NAME}_limit100.log"
OUTPUT_NAME="${NAME}_${CKPT_NAME}_qwen25_15b_gsm8k_cot_limit100"

cat > "$CONFIG_PATH" <<YAML
base_model: $MODEL_REF
peft_path: $PEFT_PATH

tasks: gsm8k_cot
limit: 100
batch_size: 1
device: cuda
dtype: bfloat16
apply_chat_template: true

output_name: $OUTPUT_NAME
YAML

python scripts/01_lmeval_eval.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"

echo "============================================================"
echo "DPO_v5 first eval 汇总"
echo "============================================================"

echo "===== flexible-extract ====="
grep -R "flexible-extract" "$MAIN_LOG_DIR"/eval_*_limit100.log || true

echo "===== strict-match ====="
grep -R "strict-match" "$MAIN_LOG_DIR"/eval_*_limit100.log || true

echo "日志目录: $MAIN_LOG_DIR"
