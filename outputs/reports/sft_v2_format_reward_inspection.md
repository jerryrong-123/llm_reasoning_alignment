# SFT small_v2 format reward inspection

## 实验目的

本报告检查从 `sft_lora_small_v2` 出发时，为什么 GRPO format-reward debug 只得到 reward=0.1 且 reward_std=0。

本脚本不训练模型，只采样生成并计算 reward 组成。

## 检查对象

```text
base_model: Qwen/Qwen2.5-1.5B-Instruct
start_adapter_path: outputs\checkpoints\sft_lora_small_v2
train_file: data\processed\grpo_format_reward_debug.jsonl
num_prompts: 3
num_generations: 4
max_new_tokens: 384
prompt_format_instruction: Final answer: <answer>
```

## Summary

| metric | value |
|---|---:|
| total_generations | 12 |
| correctness_count | 6 |
| correctness_rate | 0.5000 |
| format_hit_count | 5 |
| format_hit_rate | 0.4167 |
| extractable_count | 12 |
| extractable_rate | 1.0000 |
| avg_total_reward | 0.6417 |
| unique_reward_values | [0.1, 0.2, 1.1, 1.2000000000000002] |

## Interpretation

inspected completions 出现了不同 reward，说明通过采样和更长生成可以产生 reward 方差。

如果 format_hit 仍然很低，说明模型没有稳定遵守 `Final answer: <answer>` 格式。
如果 correctness 仍然很低，说明主要问题不是格式，而是推理或计算错误。

## Cases

| prompt_id | gen_id | gold | pred | correct | format_hit | extractable | total_reward | completion_short |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 72 | 72 | 1 | 1 | 1 | 1.2 | In April, Natalia sold 48 clips. In May, she sold half as many clips, which is 48 / 2 = <<48/2=24>>24 clips. Altogether, in both months, she sold 48 + 24 = <<48+24=72>>72 clips. Final answer: 72 |
| 0 | 1 | 72 | 72 | 1 | 1 | 1 | 1.2 | First find how many clips Natalia sold in May: 48 clips / 2 = <<48/2=24>>24 clips Then add the number of clips she sold in April and May: 24 clips + 48 clips = <<24+48=72>>72 clips  Final answer: 72 |
| 0 | 2 | 72 | 72 | 1 | 0 | 1 | 1.1 | In April, Natalia sold 48 clips. In May, Natalia sold \( \frac{1}{2} \times 48 = 24 \) clips.  Altogether, Natalia sold \( 48 + 24 = \boxed{72}\) clips in April and May. |
| 0 | 3 | 72 | 72 | 1 | 1 | 1 | 1.2 | In April, Natalia sold 48 x .5 = <<48*.5=24>>24 less clips. So in total Natalia sold in April and May = 48 + 24 = <<48+24=72>>72 #### 72 |
| 1 | 0 | 10 | 10 | 1 | 0 | 1 | 1.1 | Step 1: Determine the number of hours Weng worked. Weng worked for 50 minutes, which is equivalent to 50/60 = <<50/60=0.83>>0.83 hours.  Step 2: Calculate the earnings based on her hourly wage. Her earning per hour is $12, so for 0.83 hours... |
| 1 | 1 | 10 | 16 | 0 | 1 | 1 | 0.2 | First find how many minutes of babysitting Weng did: 50 minutes ÷ 60 minutes/hour = 8/6 hours   Then multiply her hourly wage by the number of hours to find how much she earned: 8/6 hours × $12/hour = $<<8/6*12=16>>16   #### 16  Final answe... |
| 1 | 2 | 10 | 50 | 0 | 0 | 1 | 0.1 | Step 1: Determine the hourly rate. Weng earns $12 per hour.  Step 2: Calculate how many hours were worked. Weng worked 50 minutes. To convert minutes to hours, divide by 60 (since there are 60 minutes in an hour).     50 minutes ÷ 60 = 0.83... |
| 1 | 3 | 10 | 10 | 1 | 0 | 1 | 1.1 | Step 1: Determine the rate of pay. Weng earns $12 per hour.  Step 2: Calculate how many hours Weng worked yesterday. Since there are 60 minutes in an hour, we can convert 50 minutes to hours as follows: \[ \frac{50 \text{ minutes}}{60 \text... |
| 2 | 0 | 5 | 55 | 0 | 1 | 1 | 0.2 | Betty's parents gave her $15. Her grandparents gave her 2 * $15 = $30. So far, Betty received a total of $15 + $30 = <<15+30=45>>$45. She still needs $100 - $45 = <<100-45=55>>$55.  Final answer: 55 |
| 2 | 1 | 5 | 20 | 0 | 0 | 1 | 0.1 | Betty needs 100/2 = <<100/2=50>>$ worth of savings. Her grandparents gave her 2 x $15 = <<2*$15=$30>>$30. She already saved up $50 + $30 = <<50+$30=$80>>$80. So she still needs $100 - $80 = <<100-$80=$20>>$20.  Final answer: $20 |
| 2 | 2 | 5 | 95 | 0 | 0 | 1 | 0.1 | Betty initially has 100/2 = $50. Her parents will contribute an additional 15 + (15 * 2) = $45 to get the total amount of $100 + ($50+$45) = $195. This means Betty still needs 100 - $195 = -$95 dollars.  Final answer: -$95 |
| 2 | 3 | 5 | 45 | 0 | 0 | 1 | 0.1 | Step 1: Calculate how much money Betty currently has. Betty needs $100 but she already has half of it, so: Half price = $100 / 2 = $50  Step 2: Calculate how much money Betty's parents gave her. Parents' contribution = $15  Step 3: Calculat... |
