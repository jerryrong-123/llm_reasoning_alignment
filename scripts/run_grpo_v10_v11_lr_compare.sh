#!/usr/bin/env bash
set -e

cd /root/autodl-tmp/llm_reasoning_alignment

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export HF_HUB_DISABLE_XET=1

MAIN_LOG_DIR=outputs/logs/grpo_v10_v11_lr_compare
mkdir -p "$MAIN_LOG_DIR"

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

run_train_and_eval () {
  NAME="$1"
  LR="$2"
  BETA="$3"

  OUT_DIR="outputs/checkpoints/grpo_lora_${NAME}"
  LOG_DIR="$MAIN_LOG_DIR/$NAME"
  mkdir -p "$LOG_DIR"

  echo "============================================================"
  echo "开始实验: $NAME"
  echo "learning_rate: $LR"
  echo "beta: $BETA"
  echo "output_dir: $OUT_DIR"
  echo "============================================================"

  cat > "configs/${NAME}.yaml" <<YAML
base_model: $MODEL_REF

start_adapter_path: outputs/checkpoints/sft_lora_v5_eval_style_completion_only
train_file: data/processed/grpo_v5_mixed_replay_from_sft_v5.jsonl
output_dir: $OUT_DIR

max_prompt_length: 320
max_completion_length: 384

max_steps: 50

per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: $LR

num_generations: 4
temperature: 0.7
beta: $BETA

logging_steps: 1
save_steps: 25
save_total_limit: 3

dtype: bf16
YAML

  if [ -f "$OUT_DIR/adapter_config.json" ]; then
    echo "$NAME final adapter 已存在，跳过训练。"
  else
    rm -rf "$OUT_DIR"
    python scripts/87_train_grpo_v5_from_sft_v5_server.py --config "configs/${NAME}.yaml" 2>&1 | tee "$LOG_DIR/train.log"
  fi

  CANDIDATES=()

  if [ -f "$OUT_DIR/adapter_config.json" ]; then
    CANDIDATES+=("final:$OUT_DIR")
  fi

  for CKPT in "$OUT_DIR"/checkpoint-*; do
    if [ -f "$CKPT/adapter_config.json" ]; then
      CKPT_NAME=$(basename "$CKPT")
      CANDIDATES+=("${CKPT_NAME}:${CKPT}")
    fi
  done

  echo "候选 checkpoint:"
  printf '%s\n' "${CANDIDATES[@]}"

  if [ "${#CANDIDATES[@]}" -eq 0 ]; then
    echo "没有找到可评估 checkpoint: $OUT_DIR"
    exit 1
  fi

  for ITEM in "${CANDIDATES[@]}"; do
    CKPT_NAME="${ITEM%%:*}"
    PEFT_PATH="${ITEM#*:}"
    SAFE_CKPT_NAME=$(echo "$CKPT_NAME" | sed 's/[^A-Za-z0-9_-]/_/g')

    for LIMIT in 100 200; do
      CONFIG_PATH="configs/eval_${NAME}_${SAFE_CKPT_NAME}_limit${LIMIT}.yaml"
      LOG_PATH="$LOG_DIR/eval_${SAFE_CKPT_NAME}_limit${LIMIT}.log"
      OUTPUT_NAME="${NAME}_${SAFE_CKPT_NAME}_qwen25_15b_gsm8k_cot_limit${LIMIT}"

      cat > "$CONFIG_PATH" <<YAML
base_model: $MODEL_REF
peft_path: $PEFT_PATH

tasks: gsm8k_cot
limit: $LIMIT
batch_size: 1
device: cuda
dtype: bfloat16
apply_chat_template: true

output_name: $OUTPUT_NAME
YAML

      echo "====== 评估 $NAME $CKPT_NAME limit=$LIMIT ======"
      python scripts/01_lmeval_eval.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
    done
  done
}

run_train_and_eval "grpo_v10_quick50_lr9e7_beta07" "0.0000009" "0.7"
run_train_and_eval "grpo_v11_quick50_lr11e7_beta07" "0.0000011" "0.7"

echo "============================================================"
echo "全部完成，开始汇总 flexible / strict"
echo "============================================================"

echo "===== flexible-extract ====="
grep -R "flexible-extract" "$MAIN_LOG_DIR"/*/eval_*_limit*.log || true

echo "===== strict-match ====="
grep -R "strict-match" "$MAIN_LOG_DIR"/*/eval_*_limit*.log || true

echo "日志目录: $MAIN_LOG_DIR"
