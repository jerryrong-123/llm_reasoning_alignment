# GRPO format reward inspection

## Purpose

This report inspects generated completions and reward components before running longer GRPO training.

It is used to explain why the previous GRPO debug run had:

```text
reward_mean = 0.1
reward_std = 0
loss = 0
grad_norm = 0
```

## Setup

```text
base_model: Qwen/Qwen2.5-1.5B-Instruct
dpo_adapter_path: outputs\checkpoints\dpo_lora_small
train_file: data\processed\grpo_small.jsonl
num_prompts: 3
num_generations: 2
max_new_tokens: 128
```

## Summary

| metric | value |
|---|---:|
| total_generations | 6 |
| correctness_count | 0 |
| correctness_rate | 0.0000 |
| format_hit_count | 0 |
| format_hit_rate | 0.0000 |
| extractable_count | 6 |
| extractable_rate | 1.0000 |
| avg_total_reward | 0.1000 |
| unique_reward_values | [0.1] |

## Interpretation

All inspected completions received the same reward. This explains why GRPO had no useful advantage signal.

If most outputs only receive `0.1`, they are only getting extractability reward and are missing correctness/format rewards.

## Cases

| prompt_id | gen_id | gold | pred | correct | format_hit | extractable | total_reward | completion_short |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 72 | 3 | 0 | 0 | 1 | 0.1 | Step 1: Identify the number of clips sold in April. Natalia sold clips to 48 of her friends in April.  Step 2: Determine how many clips were sold in May. Since she sold half as many clips in May compared to April, we nee... |
| 0 | 1 | 72 | 24 | 0 | 0 | 1 | 0.1 | Step 1: Identify the number of clips sold in April. Natalia sold 48 clips in April.  Step 2: Calculate the number of clips sold in May. We know that Natalia sold half as many clips in May compared to April.   Step 3: Per... |
| 1 | 0 | 10 | 36 | 0 | 0 | 1 | 0.1 | $12 x 30 = $36  Therefore, the answer is 36. |
| 1 | 1 | 10 | 3 | 0 | 0 | 1 | 0.1 | Step 1: Determine how many hours Weng worked. Since there are 60 minutes in an hour, we need to convert 50 minutes into hours. 50 minutes ÷ 60 = 0.833... (or simplified as a fraction 5/6)  Step 2: Calculate her earnings.... |
| 2 | 0 | 5 | 45 | 0 | 0 | 1 | 0.1 | Step 1: Determine how much money Betty currently has. Betty has only half of what's needed, so: $100 / 2 = $50  Step 2: Calculate how much money Betty's parents gave her. Her parents gave her: $15 (parents) x 2 = $30 (gr... |
| 2 | 1 | 5 | 4 | 0 | 0 | 1 | 0.1 | Step 1: Determine how much money Betty currently has. Betty has saved half of the $100 wallet, so: $$ \frac{1}{2} \times \$100 = \$50 $$  Step 2: Calculate the total amount given to Betty from her parents. Her parents ga... |
