# 从 SFT small_v2 出发的 GRPO format-reward 384 small 训练报告

## 实验目的

本次实验从当前最佳 checkpoint 出发：

```text
outputs/checkpoints/sft_lora_small_v2

使用 format-reward 数据和 final-answer reward，进行 5 step small-scale GRPO 训练。

本实验基于前一步 debug 结论：

max_completion_length = 256 时 reward_std = 0
max_completion_length = 384 时 reward_std = 0.5774

因此本次 small-scale 训练采用：

max_completion_length = 384
num_generations = 4
max_steps = 5

目标是验证：从 sft_lora_small_v2 出发，使用更长生成长度后，GRPO format-reward 是否可以连续多步产生有效学习信号。

配置
config: configs/grpo_format_reward_from_sft_v2_384_small.yaml
script: scripts/36_train_grpo_format_reward_from_adapter_debug.py
base_model: Qwen/Qwen2.5-1.5B-Instruct
start_adapter_path: outputs/checkpoints/sft_lora_small_v2
train_file: data/processed/grpo_format_reward_debug.jsonl
output_dir: outputs/checkpoints/grpo_lora_small_v2_format_reward_384
max_prompt_length: 320
max_completion_length: 384
max_steps: 5
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 0.00001
num_generations: 4
训练结果

训练成功完成。

adapter 已保存到：

outputs/checkpoints/grpo_lora_small_v2_format_reward_384

adapter 检查结果：

Test-Path outputs/checkpoints/grpo_lora_small_v2_format_reward_384/adapter_config.json
True

说明新的 GRPO format-reward 384 small checkpoint 已成功生成。

关键训练日志
Step 1
reward_mean: 0.600
reward_std: 0.5774
frac_reward_zero_std: 0
grad_norm: 0.7084
completions/mean_length: 384
completions/clipped_ratio: 1
Step 2
reward_mean: 0.650
reward_std: 0.5802
frac_reward_zero_std: 0
grad_norm: 0.7045
completions/mean_length: 384
completions/clipped_ratio: 1
Step 3
reward_mean: 0.100
reward_std: 0
frac_reward_zero_std: 1
grad_norm: 0
completions/mean_length: 384
completions/clipped_ratio: 1
Step 4
reward_mean: 0.375
reward_std: 0.55
frac_reward_zero_std: 0
grad_norm: 0.6637
completions/mean_length: 384
completions/clipped_ratio: 1
Step 5
reward_mean: 0.125
reward_std: 0.05
frac_reward_zero_std: 0
grad_norm: 0.9398
completions/mean_length: 384
completions/clipped_ratio: 1
结果解释

本次 5 step small-scale 训练是有效跑通的。

证据是：

1. 训练完整跑完 5 step；
2. adapter_config.json 检查为 True；
3. 多数 step 的 reward_std 大于 0；
4. 多数 step 的 grad_norm 大于 0；
5. 从 sft_lora_small_v2 出发的 GRPO format-reward 训练在 384 长度下可以产生有效 reward 方差。

和 256 debug 相比：

256 debug:
reward_std = 0
frac_reward_zero_std = 1
grad_norm = 0

384 small:
step 1 reward_std = 0.5774, grad_norm = 0.7084
step 2 reward_std = 0.5802, grad_norm = 0.7045
step 4 reward_std = 0.55,   grad_norm = 0.6637
step 5 reward_std = 0.05,   grad_norm = 0.9398

这说明提高 max_completion_length 是有效的。

仍然存在的问题

日志中仍然有一个明显问题：

completions/clipped_ratio = 1
completions/mean_length = 384

这说明模型生成仍然经常打满最大长度，可能仍然存在截断问题。

这会影响：

1. final answer line 是否完整生成；
2. format_reward 是否稳定触发；
3. correctness_reward 是否准确；
4. 后续 lm-eval strict/flexible 指标。

因此，这次训练成功并不等于最终指标一定提升。下一步必须评估新 checkpoint。

当前结论

本次实验说明：

1. 从 sft_lora_small_v2 出发做 GRPO format-reward 是可行的；
2. max_completion_length = 384 明显优于 256；
3. 新 checkpoint outputs/checkpoints/grpo_lora_small_v2_format_reward_384 已成功生成；
4. 当前训练链路具备有效 reward variance；
5. 下一步应该评估该 checkpoint，而不是继续加训练步数。
下一步方向

下一步应该创建评估配置并运行 lm-eval：

checkpoint:
outputs/checkpoints/grpo_lora_small_v2_format_reward_384

task:
gsm8k_cot

limit:
20

需要对比：

1. sft_lora_small_v2
2. grpo_lora_small
3. grpo_lora_small_format_reward
4. grpo_lora_small_v2_format_reward_384

保存后运行：

```powershell
Get-Content outputs\reports\grpo_format_reward_from_sft_v2_384_small_report.md -Encoding UTF8

Test-Path outputs\checkpoints\grpo_lora_small_v2_format_reward_384\adapter_config.json

git status --short

预期看到：

True
?? configs/grpo_format_reward_from_sft_v2_384_small.yaml
?? outputs/reports/grpo_format_reward_from_sft_v2_384_small_report.md