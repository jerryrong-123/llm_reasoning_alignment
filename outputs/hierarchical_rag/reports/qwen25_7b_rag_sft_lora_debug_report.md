# Qwen2.5-7B RAG-SFT LoRA Debug Report

## 1. Purpose

This debug run verifies that Qwen2.5-7B can be fine-tuned with 4bit QLoRA on the RAG-SFT data.

## 2. Settings

- Model: `/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct`
- Train data: `/root/autodl-tmp/llm_reasoning_alignment_server_restored/data/processed/hierarchical_rag/rag_sft_train_2000.jsonl`
- Output dir: `/root/autodl-tmp/llm_reasoning_alignment_server_restored/outputs/checkpoints/qwen25_7b_rag_sft_lora_debug`
- Train limit: 128
- Max length: 1024
- Max steps: 20
- LoRA r: 16
- LoRA alpha: 32
- Quantization: 4bit NF4

## 3. Train Metrics

- epoch: 0.625
- total_flos: 3487044272882688.0
- train_loss: 0.24056841363199055
- train_runtime: 53.3009
- train_samples_per_second: 1.501
- train_steps_per_second: 0.375
