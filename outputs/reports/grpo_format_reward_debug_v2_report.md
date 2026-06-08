# GRPO format-reward debug v2 运行报告

## 实验目的

本次实验是在第一次 GRPO format-reward debug 的基础上继续改进。

第一次 debug 已经证明：

```text
final-answer reward 可以接入 GRPOTrainer；
adapter 可以正常保存；
但是 reward_std = 0，loss = 0，grad_norm = 0。
```

这说明第一次只是链路跑通，还没有产生有效学习信号。

因此本次 v2 debug 做了三个改动：

```text
1. 使用 format-reward 专用 GRPO 数据；
2. 将 num_generations 从 2 提升到 4；
3. 将 max_completion_length 从 128 提升到 256。
```

目标是验证：GRPO 是否能看到 reward 方差，从而产生有效训练信号。

## 新增数据

新增数据由脚本生成：

```text
scripts/33_prepare_grpo_format_reward_data.py
```

输入数据：

```text
data/processed/grpo_small.jsonl
```

输出数据：

```text
data/processed/grpo_format_reward_debug.jsonl
data/samples/grpo_format_reward_debug_preview.jsonl
```

新数据在每个 prompt 后追加格式要求：

```text
After solving, end your response with exactly one final line:
Final answer: <answer>
```

这样做的原因是：reward inspection v2 已经证明，更明确的格式提示可以让模型产生更多不同 reward 的样本。

## 配置

本次运行使用：

```text
config: configs/grpo_format_reward_debug.yaml
script: scripts/31_train_grpo_format_reward_debug.py
base_model: Qwen/Qwen2.5-1.5B-Instruct
dpo_adapter_path: outputs/checkpoints/dpo_lora_small
train_file: data/processed/grpo_format_reward_debug.jsonl
output_dir: outputs/checkpoints/grpo_lora_small_format_reward_debug_v2
max_prompt_length: 320
max_completion_length: 256
max_steps: 1
per_device_train_batch_size: 4
num_generations: 4
```

## 运行结果

训练成功完成。

adapter 已保存到：

```text
outputs/checkpoints/grpo_lora_small_format_reward_debug_v2
```

adapter 检查结果：

```text
Test-Path outputs/checkpoints/grpo_lora_small_format_reward_debug_v2/adapter_config.json
True
```

## 关键训练日志

```text
loss: -8.941e-08
grad_norm: 0.7355
learning_rate: 1e-05
num_tokens: 1432
completions/mean_length: 256
completions/min_length: 256
completions/max_length: 256
completions/clipped_ratio: 1
rewards/final_answer_reward_func/mean: 0.6
rewards/final_answer_reward_func/std: 0.5774
reward: 0.6
reward_std: 0.5774
frac_reward_zero_std: 0
entropy: 0.3017
train_runtime: 121.8
train_loss: -8.941e-08
```

## 对比 v1 debug

| item                  |         v1 debug |                       v2 debug |
| --------------------- | ---------------: | -----------------------------: |
| train_file            | grpo_small.jsonl | grpo_format_reward_debug.jsonl |
| max_completion_length |              128 |                            256 |
| num_generations       |                2 |                              4 |
| reward_mean           |              0.1 |                            0.6 |
| reward_std            |                0 |                         0.5774 |
| frac_reward_zero_std  |                1 |                              0 |
| grad_norm             |                0 |                         0.7355 |

## 结果解释

v2 是一个明显更好的 GRPO debug 结果。

第一次 debug 的问题是：

```text
所有 sampled completions 的 reward 都是 0.1；
reward_std = 0；
GRPO 没有 advantage signal；
loss 和 grad_norm 都是 0。
```

本次 v2 中：

```text
reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7355
```

说明同一个 prompt 下的多个 generations 已经拿到了不同 reward，GRPO 可以进行相对比较。

这意味着当前 final-answer reward 不只是能接入训练链路，而且已经能提供有效学习信号。

## 当前结论

本次结果证明：

```text
1. format-reward 专用 prompt 数据是必要的；
2. num_generations=4 比 num_generations=2 更适合这个 debug；
3. max_completion_length=256 能减少过早截断；
4. reward_std 已经大于 0；
5. 当前设置可以作为后续 small-scale GRPO format reward 训练的起点。
```

## 下一步方向

下一步可以从 debug 进入小步数训练，但仍然不能直接大规模训练。

建议下一阶段：

```text
1. 将 max_steps 从 1 提升到 5 或 10；
2. 保持 num_generations = 4；
3. 保持 max_completion_length = 256；
4. 训练后评估新的 checkpoint；
5. 对比 sft_lora_small_v2、grpo_lora_small、grpo_lora_small_format_reward_debug_v2。
```

注意：如果后续训练仍然出现 completion 全部被截断，则需要继续提高 max_completion_length 或缩短 prompt。
