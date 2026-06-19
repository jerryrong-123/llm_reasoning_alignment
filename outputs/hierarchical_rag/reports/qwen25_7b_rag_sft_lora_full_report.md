# Qwen2.5-7B RAG-SFT LoRA Full Report

## 1. Purpose

This full run fine-tunes Qwen2.5-7B with 4bit QLoRA on the RAG-SFT data.

## 2. Settings

- Model: `/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct`
- Train data: `/root/autodl-tmp/llm_reasoning_alignment_server_restored/data/processed/hierarchical_rag/rag_sft_train_2000.jsonl`
- Output dir: `/root/autodl-tmp/llm_reasoning_alignment_server_restored/outputs/checkpoints/qwen25_7b_rag_sft_lora_full`
- Train limit: 2000
- Max length: 1024
- Max steps: 500
- LoRA r: 16
- LoRA alpha: 32
- Quantization: 4bit NF4

## 3. Train Metrics

- epoch: 1.0
- total_flos: 8.659363146554266e+16
- train_loss: 0.1569909765558532
- train_runtime: 1306.2422
- train_samples_per_second: 1.531
- train_steps_per_second: 0.383
