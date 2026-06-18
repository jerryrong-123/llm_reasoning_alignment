# 从 SFT small_v2 出发的 GRPO format-reward debug 报告

## 实验目的

本次实验测试 GRPO format-reward 训练是否可以从当前最佳 checkpoint 出发：

```text
outputs/checkpoints/sft_lora_small_v2

之前的 GRPO format-reward 分支是从：

outputs/checkpoints/dpo_lora_small

出发的。

但是前面的评估已经说明：

grpo_lora_small_format_reward 没有超过 sft_lora_small_v2

所以本次尝试从当前更强的 sft_lora_small_v2 出发，看是否能产生更好的 reward 学习信号。

配置

本次运行使用：

config: configs/grpo_format_reward_from_sft_v2_debug.yaml
script: scripts/36_train_grpo_format_reward_from_adapter_debug.py
base_model: Qwen/Qwen2.5-1.5B-Instruct
start_adapter_path: outputs/checkpoints/sft_lora_small_v2
train_file: data/processed/grpo_format_reward_debug.jsonl
output_dir: outputs/checkpoints/grpo_lora_small_v2_format_reward_debug
max_prompt_length: 320
max_completion_length: 256
max_steps: 1
per_device_train_batch_size: 4
num_generations: 4
运行结果

训练链路成功跑通。

adapter 已保存到：

outputs/checkpoints/grpo_lora_small_v2_format_reward_debug

adapter 检查结果：

Test-Path outputs/checkpoints/grpo_lora_small_v2_format_reward_debug/adapter_config.json
True

这说明新的通用脚本可以从 sft_lora_small_v2 加载 LoRA adapter，运行 GRPOTrainer，并保存新的 LoRA adapter。

关键训练日志
loss: 0
grad_norm: 0
learning_rate: 1e-05
num_tokens: 1432
completions/mean_length: 256
completions/min_length: 256
completions/max_length: 256
completions/clipped_ratio: 1
rewards/final_answer_reward_func/mean: 0.1
rewards/final_answer_reward_func/std: 0
reward: 0.1
reward_std: 0
frac_reward_zero_std: 1
entropy: 0.3972
train_runtime: 128.8
train_loss: 0
结果解释

这次实验是一次成功的链路 debug，但不是一次成功的学习实验。

关键问题是：

reward_std = 0
frac_reward_zero_std = 1
loss = 0
grad_norm = 0

这说明同一组 sampled completions 得到了完全一样的 reward。

本次平均 reward 是：

0.1

这大概率说明模型生成结果只拿到了 extractability_reward，没有拿到：

correctness_reward
format_reward

因此，GRPO 没有可比较的 advantage signal。

和 DPO-start debug v2 对比

之前从 dpo_lora_small 出发的 GRPO format-reward debug v2 有有效 reward 方差：

start_adapter_path: outputs/checkpoints/dpo_lora_small
reward_mean: 0.6
reward_std: 0.5774
frac_reward_zero_std: 0
grad_norm: 0.7355

本次从 sft_lora_small_v2 出发的 debug 没有 reward 方差：

start_adapter_path: outputs/checkpoints/sft_lora_small_v2
reward_mean: 0.1
reward_std: 0
frac_reward_zero_std: 1
grad_norm: 0

所以，在当前 prompt、当前数据、当前 max_completion_length=256、当前 num_generations=4 的设置下，从 sft_lora_small_v2 出发并不会自动产生更好的 GRPO format-reward 学习信号。

当前结论

本次实验证明：

1. 通用 adapter-start GRPO format-reward 脚本可以正常工作；
2. GRPO 可以从 sft_lora_small_v2 出发；
3. 新 adapter 可以成功保存；
4. 但当前设置下 reward_std = 0；
5. 这个配置不应该直接扩大到更多训练步数。

当前最重要的结论是：

从最佳 SFT checkpoint 出发是合理方向，
但当前 prompt / generation / reward 设置还不能产生有效 reward 方差。

所以现在不能直接把：

max_steps: 1

加到：

max_steps: 5 或 10

否则只是浪费时间。

下一步方向

下一步不应该继续训练，而应该先做 sample inspection。

建议下一步：

新增一个专门针对 sft_lora_small_v2 的 reward inspection 脚本。

它需要打印：

prompt
gold answer
generated completion
extracted prediction
correctness_reward
format_reward
extractability_reward
total_reward

这样才能判断为什么从 sft_lora_small_v2 出发时，所有 sampled completions 都只有 0.1 reward。