# 从 SFT small_v2 出发的 GRPO format-reward debug 384 报告

## 实验目的

本次实验继续检查从当前最佳 checkpoint 出发的 GRPO format-reward 训练：

```text
outputs/checkpoints/sft_lora_small_v2

上一次 max_completion_length = 256 的 debug 虽然能保存 adapter，但没有有效学习信号：

reward_std = 0
frac_reward_zero_std = 1
grad_norm = 0
loss = 0

随后 inspection 发现，当生成长度提高到 384 时，sft_lora_small_v2 可以产生不同 reward。

因此本次实验把 GRPO debug 配置改成：

max_completion_length = 384

目标是验证：在真实 GRPOTrainer 里，384 长度是否也能产生 reward 方差。

配置

本次运行使用：

config: configs/grpo_format_reward_from_sft_v2_debug_384.yaml
script: scripts/36_train_grpo_format_reward_from_adapter_debug.py
base_model: Qwen/Qwen2.5-1.5B-Instruct
start_adapter_path: outputs/checkpoints/sft_lora_small_v2
train_file: data/processed/grpo_format_reward_debug.jsonl
output_dir: outputs/checkpoints/grpo_lora_small_v2_format_reward_debug_384
max_prompt_length: 320
max_completion_length: 384
max_steps: 1
per_device_train_batch_size: 4
num_generations: 4
运行结果

训练成功完成。

adapter 已保存到：

outputs/checkpoints/grpo_lora_small_v2_format_reward_debug_384

adapter 检查结果：

Test-Path outputs/checkpoints/grpo_lora_small_v2_format_reward_debug_384/adapter_config.json
True
关键训练日志

本次终端日志中部分字段显示被截断，但关键字段可读：

loss: 0
grad_norm: 0.7084
learning_rate: 1e-05
num_tokens: 1944
completions/mean_length: 384
completions/min_length: 384
completions/max_length: 384
completions/clipped_ratio: 1
reward_std: 0.5774
frac_reward_zero_std: 0
entropy: 0.3735
train_runtime: 179.9
train_loss: 0
和 256 debug 对比
SFT-v2 start, max_completion_length = 256:
reward_mean = 0.1
reward_std = 0
frac_reward_zero_std = 1
grad_norm = 0

SFT-v2 start, max_completion_length = 384:
reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7084
结果解释

这次 384 debug 是有效的。

关键变化是：

reward_std 从 0 变成 0.5774
frac_reward_zero_std 从 1 变成 0
grad_norm 从 0 变成 0.7084

这说明同一组 sampled completions 已经拿到了不同 reward，GRPO 有了可比较的 advantage signal。

但是仍然有一个问题：

completions/clipped_ratio = 1
completions/mean_length = 384

这说明模型仍然经常打满最大生成长度，后续可能还存在截断问题。

当前结论

本次实验说明：

1. 从 sft_lora_small_v2 出发做 GRPO format reward 是可行的；
2. max_completion_length = 256 不够，容易导致 reward_std = 0；
3. max_completion_length = 384 可以产生有效 reward 方差；
4. 当前配置可以作为后续 small-scale GRPO from SFT-v2 的起点；
5. 但后续仍要关注 completions/clipped_ratio = 1 的截断问题。
下一步方向

下一步可以创建 small-scale 配置：

configs/grpo_format_reward_from_sft_v2_384_small.yaml

建议参数：

start_adapter_path: outputs/checkpoints/sft_lora_small_v2
max_completion_length: 384
num_generations: 4
max_steps: 5
output_dir: outputs/checkpoints/grpo_lora_small_v2_format_reward_384

然后再评估新 checkpoint 是否超过：

sft_lora_small_v2
grpo_lora_small
grpo_lora_small_format_reward

保存后运行：

```powershell
Get-Content outputs\reports\grpo_format_reward_from_sft_v2_debug_384_report.md -Encoding UTF8

git status --short

确认看到：

?? configs/grpo_format_reward_from_sft_v2_debug_384.yaml
?? outputs/reports/grpo_format_reward_from_sft_v2_debug_384_report.md