# GRPO format-reward debug 运行报告

## 实验目的

本次 debug run 用来验证新加入的 final-answer reward 是否可以成功接入现有 GRPO/RLVR 训练链路。

这不是正式训练，只是一个 `max_steps = 1` 的链路验证实验。

## 配置

```text
config: configs/grpo_format_reward_debug.yaml
script: scripts/31_train_grpo_format_reward_debug.py
base_model: Qwen/Qwen2.5-1.5B-Instruct
dpo_adapter_path: outputs/checkpoints/dpo_lora_small
train_file: data/processed/grpo_small.jsonl
output_dir: outputs/checkpoints/grpo_lora_small_format_reward_debug
max_steps: 1
num_generations: 2
```

## Reward 函数

本次运行使用：

```text
src/rewards/final_answer_reward.py
```

Reward 设计为：

```text
correctness_reward:
  +1.0 if extracted numeric answer equals gold answer

format_reward:
  +0.1 if response contains a clear final-answer line

extractability_reward:
  +0.1 if a numeric prediction can be extracted

total_reward = correctness_reward + format_reward + extractability_reward
```

核心原则是：

```text
answer correctness reward > final-answer format reward
```

这样可以避免模型只学会输出格式，但答案仍然错误。

## 运行结果

本次 GRPO format-reward debug run 已经成功完成。

adapter 已保存到：

```text
outputs/checkpoints/grpo_lora_small_format_reward_debug
```

adapter 检查结果为：

```text
Test-Path outputs/checkpoints/grpo_lora_small_format_reward_debug/adapter_config.json
True
```

这说明 final-answer reward 已经成功接入 GRPOTrainer 链路。

## 关键训练日志

```text
loss: 0
grad_norm: 0
learning_rate: 1e-05
num_tokens: 424
completions/mean_length: 128
completions/clipped_ratio: 1
rewards/final_answer_reward_func/mean: 0.1
rewards/final_answer_reward_func/std: 0
reward: 0.1
reward_std: 0
frac_reward_zero_std: 1
entropy: 0.4035
train_runtime: 43.47
train_loss: 0
```

## 结果解释

本次实验是一次成功的链路 debug，但还不是一次有效的学习实验。

关键问题是：

```text
reward_std = 0
frac_reward_zero_std = 1
loss = 0
grad_norm = 0
```

这说明同一组 sampled completions 拿到的 reward 完全一样，GRPO 没有可比较的优势信号。

本次平均 reward 只有：

```text
0.1
```

这大概率说明 sampled completions 只拿到了 extractability reward，没有拿到 correctness reward，也没有拿到严格 final-answer format reward。

## 当前结论

本次结果证明：

```text
1. final-answer reward 模块可以正常工作；
2. reward 函数可以被 GRPOTrainer 调用；
3. GRPO format-reward debug 脚本可以训练并保存 adapter；
4. 但当前 1 step run 没有产生有效 reward 方差。
```

所以，下一步不应该盲目增加 `max_steps`。

更合理的下一步是先检查模型实际生成内容和每个 reward 组成部分，找出为什么 reward 全是 `0.1`。

## 下一步方向

可能的改进方向：

```text
1. 增加 num_generations，例如从 2 改到 4；
2. 增加 max_completion_length，减少回答被截断；
3. 加入更细粒度的部分格式 reward；
4. 打印生成样本和 reward 组成；
5. 在正式增加训练步数前，先做 reward inspection。
```

推荐下一步：

```text
新增 GRPO reward inspection 脚本。
```

这个脚本应该输出：

```text
prompt
gold answer
generated completion
extracted prediction
correctness_reward
format_reward
extractability_reward
total_reward
```

这样才能判断为什么当前 reward mean 只有 `0.1`，以及为什么 reward variance 是 0。