#!/usr/bin/env bash
set -e

cd /root/autodl-tmp/llm_reasoning_alignment

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export HF_HUB_DISABLE_XET=1

LOG_DIR=outputs/logs
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
save_total_limit: 3

dtype: bf16
YAML

echo "====== 开始训练 GRPO_v9 quick75 ======"
if [ -f outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative/adapter_config.json ]; then
  echo "GRPO_v9 checkpoint 已存在，跳过训练。"
else
  rm -rf outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative
  python scripts/87_train_grpo_v5_from_sft_v5_server.py --config configs/grpo_v9_quick75_stronger_conservative.yaml 2>&1 | tee "$LOG_DIR/grpo_v9_train.log"
fi

echo "====== 创建并运行 limit=100 eval ======"
cat > configs/eval_grpo_v9_quick75_stronger_conservative_limit100.yaml <<YAML
base_model: $MODEL_REF
peft_path: outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative

tasks: gsm8k_cot
limit: 100
batch_size: 1
device: cuda
dtype: bfloat16
apply_chat_template: true

output_name: grpo_lora_v9_quick75_stronger_conservative_qwen25_15b_gsm8k_cot_limit100
YAML

python scripts/01_lmeval_eval.py --config configs/eval_grpo_v9_quick75_stronger_conservative_limit100.yaml 2>&1 | tee "$LOG_DIR/grpo_v9_eval_limit100.log"

echo "====== 创建并运行 limit=200 eval ======"
cat > configs/eval_grpo_v9_quick75_stronger_conservative_limit200.yaml <<YAML
base_model: $MODEL_REF
peft_path: outputs/checkpoints/grpo_lora_v9_quick75_stronger_conservative

tasks: gsm8k_cot
limit: 200
batch_size: 1
device: cuda
dtype: bfloat16
apply_chat_template: true

output_name: grpo_lora_v9_quick75_stronger_conservative_qwen25_15b_gsm8k_cot_limit200
YAML

python scripts/01_lmeval_eval.py --config configs/eval_grpo_v9_quick75_stronger_conservative_limit200.yaml 2>&1 | tee "$LOG_DIR/grpo_v9_eval_limit200.log"

echo "====== 全部完成 ======"
echo "训练日志: $LOG_DIR/grpo_v9_train.log"
echo "limit100 日志: $LOG_DIR/grpo_v9_eval_limit100.log"
echo "limit200 日志: $LOG_DIR/grpo_v9_eval_limit200.log"
