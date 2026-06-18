# GRPO format-reward small 训练报告

## 实验目的

本次实验是在 `GRPO format-reward debug v2` 成功之后，进一步进行 small-scale GRPO format reward 训练。

前一步 debug v2 已经证明：

```text
1. final-answer reward 可以接入 GRPOTrainer；
2. 使用 format-reward 专用 prompt 数据后，reward 不再恒定；
3. num_generations = 4 和 max_completion_length = 256 可以产生 reward 方差；
4. GRPO 已经有有效学习信号。
```

因此本次实验将训练步数从 debug 阶段的：

```text
max_steps = 1
```

提升到：

```text
max_steps = 5
```

目标是验证：在本地 CPU 环境下，format-reward GRPO 是否可以进行小规模连续训练，并成功保存新的 LoRA adapter。

## 配置

本次运行使用：

```text
config: configs/grpo_format_reward_small.yaml
script: scripts/31_train_grpo_format_reward_debug.py
base_model: Qwen/Qwen2.5-1.5B-Instruct
dpo_adapter_path: outputs/checkpoints/dpo_lora_small
train_file: data/processed/grpo_format_reward_debug.jsonl
output_dir: outputs/checkpoints/grpo_lora_small_format_reward
max_prompt_length: 320
max_completion_length: 256
max_steps: 5
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.00001
num_generations: 4
```

## 输入数据

训练数据来自：

```text
data/processed/grpo_format_reward_debug.jsonl
```

该数据由以下脚本生成：

```text
scripts/33_prepare_grpo_format_reward_data.py
```

数据构造逻辑：

```text
基于 data/processed/grpo_small.jsonl，
在每条 prompt 后追加明确的最终答案格式要求：
After solving, end your response with exactly one final line:
Final answer: <answer>
```

这样做是为了让模型在 GRPO 采样时更容易触发 format reward，同时仍然让 correctness reward 作为主 reward。

## Reward 设计

本次训练使用：

```text
src/rewards/final_answer_reward.py
```

reward 结构是：

```text
correctness_reward:
  +1.0 if extracted numeric answer equals gold answer

format_reward:
  +0.1 if response contains a clear final-answer line

extractability_reward:
  +0.1 if a numeric prediction can be extracted

total_reward = correctness_reward + format_reward + extractability_reward
```

核心原则：

```text
answer correctness reward > final-answer format reward
```

也就是说，格式只是辅助奖励，不能压过答案正确性奖励。

## 训练结果

训练成功完成。

新的 adapter 已保存到：

```text
outputs/checkpoints/grpo_lora_small_format_reward
```

adapter 检查结果：

```text
Test-Path outputs/checkpoints/grpo_lora_small_format_reward/adapter_config.json
True
```

说明本次 small-scale GRPO format reward 训练已经成功生成 checkpoint。

## 关键训练日志

### Step 1

```text
loss: -8.941e-08
grad_norm: 0.7355
learning_rate: 1e-05
reward_mean: 0.6
reward_std: 0.5774
frac_reward_zero_std: 0
completions/mean_length: 256
completions/clipped_ratio: 1
```

### Step 2

```text
loss: 1.192e-07
grad_norm: 1.584
learning_rate: 8e-06
reward_mean: 0.4
reward_std: 0.5354
frac_reward_zero_std: 0
completions/mean_length: 256
completions/clipped_ratio: 1
```

### Step 3

```text
loss: 0
grad_norm: 0
learning_rate: 6e-06
reward_mean: 0.1
reward_std: 0
frac_reward_zero_std: 1
completions/mean_length: 256
completions/clipped_ratio: 1
```

### Step 4

```text
loss: -5.96e-08
grad_norm: 2.829
learning_rate: 4e-06
reward_mean: 0.35
reward_std: 0.5
frac_reward_zero_std: 0
completions/mean_length: 256
completions/clipped_ratio: 1
```

### Step 5

```text
loss: -1.49e-08
grad_norm: 0.788
learning_rate: 2e-06
reward_mean: 0.125
reward_std: 0.05
frac_reward_zero_std: 0
completions/mean_length: 256
completions/clipped_ratio: 1
```

## 结果解释

本次 5 step small-scale GRPO format reward 训练是成功的。

主要证据是：

```text
1. 训练完整跑完 5 step；
2. adapter_config.json 检查为 True；
3. 多数 step 的 reward_std 大于 0；
4. 多数 step 的 grad_norm 大于 0；
5. GRPO 在多个 step 中获得了有效 reward variance。
```

和第一次 format-reward debug 相比：

```text
第一次 debug:
reward_mean = 0.1
reward_std = 0
frac_reward_zero_std = 1
grad_norm = 0

本次 small-scale:
step 1 reward_std = 0.5774, grad_norm = 0.7355
step 2 reward_std = 0.5354, grad_norm = 1.584
step 4 reward_std = 0.5,    grad_norm = 2.829
step 5 reward_std = 0.05,   grad_norm = 0.788
```

这说明 format-reward 专用数据、`num_generations = 4`、`max_completion_length = 256` 的组合是有效的。

## 仍然存在的问题

虽然训练成功，但日志里仍有一个明显问题：

```text
completions/clipped_ratio = 1
completions/mean_length = 256
```

这说明模型生成几乎每次都打满 `max_completion_length = 256`，回答仍然可能被截断。

这会影响：

```text
1. final answer line 是否能完整生成；
2. format_reward 是否稳定触发；
3. correctness extraction 是否准确；
4. 后续评估时的 strict / format 指标。
```

因此，后续如果继续优化，可能需要：

```text
1. 提高 max_completion_length；
2. 缩短 prompt；
3. 让模型更早输出 Final answer；
4. 在 reward 中加入对过长输出或未终止输出的轻微惩罚。
```

## 当前结论

本次实验可以作为项目中的一个正向 small-scale RLVR/GRPO 结果：

```text
1. reward-based format optimization 已经从 debug 进入 small-scale training；
2. final-answer reward 可以在 GRPO 中产生有效 reward variance；
3. 新 checkpoint outputs/checkpoints/grpo_lora_small_format_reward 已成功生成；
4. 下一步应该评估该 checkpoint，而不是继续盲目训练。
```

## 下一步方向

下一步应该创建该 checkpoint 的评估配置，并运行 lm-eval：

```text
checkpoint:
outputs/checkpoints/grpo_lora_small_format_reward

建议评估：
GSM8K-COT limit=20

需要对比：
1. sft_lora_small_v2
2. dpo_lora_small
3. grpo_lora_small
4. grpo_lora_small_format_reward
```

如果评估结果显示 flexible 或 strict 没有提升，也仍然可以作为一个有价值的负结果或中间结果，因为它证明了：

```text
reward 接入成功；
reward variance 成功产生；
small-scale GRPO 训练成功；
但最终指标是否提升需要进一步评估确认。
```
