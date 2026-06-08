# GRPO format reward inspection

## Purpose

This report inspects generated completions and reward components before running longer GRPO training.

This v2 inspection uses a stronger final-answer instruction and longer generation length.

## Setup

```text
base_model: Qwen/Qwen2.5-1.5B-Instruct
dpo_adapter_path: outputs\checkpoints\dpo_lora_small
train_file: data\processed\grpo_small.jsonl
num_prompts: 3
num_generations: 4
max_new_tokens: 256
prompt_format_instruction: Final answer: <answer>
```

## Summary

| metric | value |
|---|---:|
| total_generations | 12 |
| correctness_count | 7 |
| correctness_rate | 0.5833 |
| format_hit_count | 6 |
| format_hit_rate | 0.5000 |
| extractable_count | 12 |
| extractable_rate | 1.0000 |
| avg_total_reward | 0.7333 |
| unique_reward_values | [0.1, 0.2, 1.1, 1.2000000000000002] |

## Interpretation

The inspected completions received different rewards. This means the reward function can produce useful variance under sampling.

If format_hit remains 0, the model is not following the explicit final-answer format even with a stronger prompt.
If correctness remains 0, the main bottleneck is reasoning/answer correctness rather than formatting.

## Cases

| prompt_id | gen_id | gold | pred | correct | format_hit | extractable | total_reward | completion_short |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 72 | 72 | 1 | 1 | 1 | 1.2 | Step 1: Determine how many clips Natalia sold in May. Natalia sold half as many clips in May as she did in April. Since she sold 48 clips in April, we calculate half of 48. \[ \text{Clips sold in May} = \frac{48}{2} \]  Step 2: Perform the ... |
| 0 | 1 | 72 | 72 | 1 | 1 | 1 | 1.2 | Step 1: Determine how many clips Natalia sold in May. Natalia sold half as many clips in May as she did in April.  Step 2: Calculate the number of clips sold in May. Number of clips sold in May = Number of clips sold in April ÷ 2  Step 3: A... |
| 0 | 2 | 72 | 72 | 1 | 1 | 1 | 1.2 | Step 1: Determine how many clips Natalia sold in May. - Natalia sold 48 clips in April. - In May, she sold half as many clips as she did in April.  Step 2: Calculate half of the number of clips sold in April. - To find out how many clips Na... |
| 0 | 3 | 72 | 72 | 1 | 1 | 1 | 1.2 | Step 1: Determine how many clips Natalia sold in May. Natalia sold half as many clips in May as she did in April, so we need to divide the number of clips sold in April by 2. Number of clips sold in May = Number of clips sold in April / 2  ... |
| 1 | 0 | 10 | 10 | 1 | 0 | 1 | 1.1 | Step 1: Determine the total number of hours Weng worked. Weng worked 50 minutes of babysitting yesterday. To convert minutes to hours, we divide by 60 since there are 60 minutes in an hour: \[ \text{Hours} = \frac{\text{Minutes}}{60} = \fra... |
| 1 | 1 | 10 | 10 | 1 | 0 | 1 | 1.1 | Step 1: Determine the time Weng spent on babysitting. Weng spent 50 minutes on babysitting yesterday.  Step 2: Convert minutes to hours since her hourly rate is given in dollars per hour. There are 60 minutes in one hour. \[ \frac{50 \text{... |
| 1 | 2 | 10 | 10 | 1 | 0 | 1 | 1.1 | Step 1: Determine the rate of pay. Weng earns $12 per hour.  Step 2: Convert minutes to hours because the rate is given in hours. There are 60 minutes in an hour. So we need to convert 50 minutes to hours: \[ \text{Time in hours} = \frac{\t... |
| 1 | 3 | 10 | 5 | 0 | 0 | 1 | 0.1 | Step 1: Determine the fraction of an hour that 50 minutes represents. There are 60 minutes in an hour, so we calculate: \[ \text{Fraction} = \frac{\text{Minutes}}{\text{Total minutes in an hour}} = \frac{50}{60} \]  Step 2: Simplify the fra... |
| 2 | 0 | 5 | 55 | 0 | 1 | 1 | 0.2 | Step 1: Calculate how much money Betty currently has. Betty already has half of what's needed, so we calculate: Half cost = $100 ÷ 2 = $50  Step 2: Figure out how much money her parents gave her. Her parents contributed $15.  Step 3: Determ... |
| 2 | 1 | 5 | 20 | 0 | 0 | 1 | 0.1 | Step 1: Calculate how much Betty currently has. Betty needs $100 for the wallet but only has half of it because: \[ \text{Amount needed} = \$100 \] \[ \frac{\text{Amount needed}}{2} = \frac{\$100}{2} = \$50 \]  Step 2: Calculate how much he... |
| 2 | 2 | 5 | 10 | 0 | 0 | 1 | 0.1 | Step 1: Calculate how much money Betty already has. Betty has half of what's needed, so: Half of $100 = $100 / 2 = $50  Step 2: Add up all the amounts given. Her grandparents gave her $15, and her parents gave her $15 + $30 (twice as much) ... |
| 2 | 3 | 5 | 0 | 0 | 1 | 1 | 0.2 | Step 1: Calculate the amount of money Betty currently has. Betty needs $100 but has only half of that amount. So, Betty currently has: $100 / 2 = $50  Step 2: Add the amounts given by Betty's parents and grandparents. Her parents gave her $... |
