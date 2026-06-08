# GRPO format-reward small 评估报告

## 实验目的

本报告记录 `grpo_lora_small_format_reward` 的 lm-eval 评估结果。

该 checkpoint 来自 reward-based format optimization 小规模训练：

```text
checkpoint:
outputs/checkpoints/grpo_lora_small_format_reward
```

训练阶段已经证明：

```text
1. final-answer reward 可以接入 GRPOTrainer；
2. format-reward 专用数据可以产生 reward 方差；
3. small-scale GRPO format reward 训练可以成功保存 LoRA adapter；
4. 训练日志中多个 step 出现 reward_std > 0 和 grad_norm > 0。
```

但是训练链路成功不等于最终任务指标提升，所以必须继续做 lm-eval。

## 评估配置

本次评估使用：

```text
config:
configs/eval_grpo_format_reward_lora.yaml
```

配置内容：

```text
base_model: Qwen/Qwen2.5-1.5B-Instruct
peft_path: outputs/checkpoints/grpo_lora_small_format_reward
tasks: gsm8k_cot
limit: 20
batch_size: 1
device: cpu
dtype: float32
apply_chat_template: true
output_name: grpo_lora_small_format_reward_qwen25_15b_gsm8k_cot_limit20
```

运行命令：

```powershell
python scripts\01_lmeval_eval.py --config configs\eval_grpo_format_reward_lora.yaml
```

评估输出目录：

```text
outputs/eval/grpo_lora_small_format_reward_qwen25_15b_gsm8k_cot_limit20
```

## 评估结果

lm-eval 输出结果：

```text
| Task      | Filter           | n-shot | Metric      | Value | Stderr |
|-----------|------------------|-------:|-------------|------:|-------:|
| gsm8k_cot | flexible-extract |      8 | exact_match | 0.4000 | 0.1124 |
| gsm8k_cot | strict-match     |      8 | exact_match | 0.2000 | 0.0918 |
```

因此：

```text
grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000
```

## 和已有结果对比

当前关键实验结果：

```text
sft_lora_small:
flexible = 0.4500
strict   = 0.2500

dpo_lora_small:
flexible = 0.4000
strict   = 0.2000

grpo_lora_small:
flexible = 0.4000
strict   = 0.2000

sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000

sft_lora_small_v2_format:
flexible = 0.3500
strict   = 0.1500

prompt-level format eval v1:
flexible_acc    = 0.4000
strict_hash_acc = 0.4000

prompt-level format eval v2:
flexible_acc = 0.5500
format_acc   = 0.4000

grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000
```

## 结果解释

这次结果说明：

```text
1. reward-based format optimization 的训练链路是成功的；
2. final-answer reward 能产生 reward variance；
3. small-scale GRPO format reward checkpoint 已经生成；
4. 但是在 GSM8K-COT limit=20 上，最终指标没有超过当前最佳 sft_lora_small_v2。
```

和当前最佳结果相比：

```text
sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000

grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000
```

也就是说：

```text
flexible: 0.6000 -> 0.4000，下降 0.2000
strict:   0.2000 -> 0.2000，持平
```

这说明当前 5 step GRPO format reward 训练没有改善 GSM8K-COT 最终任务表现。

## 为什么训练成功但评估没有提升

可能原因如下：

```text
1. max_steps = 5 太少，只是 small-scale 验证，还不足以稳定提升任务指标；
2. completions/clipped_ratio = 1，说明生成经常被 max_completion_length 截断；
3. reward 中 format_reward 只有 0.1，对 lm-eval strict-match 的直接提升有限；
4. 当前 reward 主要验证 final-answer 格式和可抽取性，没有直接解决 reasoning_or_calc_error；
5. GRPO 是从 dpo_lora_small 接着训练，而当前最佳 checkpoint 是 sft_lora_small_v2。
```

尤其第 5 点很重要：

```text
grpo_lora_small_format_reward 是从 dpo_lora_small 出发；
但当前最佳模型是 sft_lora_small_v2。
```

因此，这次结果不能说明 reward-based format optimization 完全无效，只能说明：

```text
在当前起点、当前数据、当前 5 step CPU 小规模训练下，没有超过 sft_lora_small_v2。
```

## 当前结论

本次实验是一个有价值的中间结果，但不是最终正向指标结果。

它证明了：

```text
1. final-answer reward 可以被 GRPO 使用；
2. format-reward 数据构造是有效的；
3. GRPO small-scale 训练可以产生 reward variance；
4. 训练 checkpoint 可以生成并被 lm-eval 加载；
5. 但最终 GSM8K-COT 指标没有提升。
```

因此，当前最强结论应该写成：

```text
reward-based format optimization pipeline works,
but the current 5-step GRPO format-reward run does not improve GSM8K-COT accuracy.
```

中文表述：

```text
reward-based format optimization 链路已经跑通，
但当前 5 step GRPO format-reward 训练没有带来 GSM8K-COT 指标提升。
```

## 下一步方向

下一步不应该盲目继续加训练步数。

更合理的方向是：

```text
1. 先收集 grpo_lora_small_format_reward 的样本输出；
2. 对比 sft_lora_small_v2、grpo_lora_small、grpo_lora_small_format_reward 的 bad cases；
3. 判断 GRPO format reward 到底破坏了哪些题；
4. 如果继续做 RLVR，应考虑从 sft_lora_small_v2 出发，而不是从 dpo_lora_small 出发；
5. 同时需要解决 completions/clipped_ratio = 1 的截断问题。
```

下一步建议：

```text
新增 grpo_lora_small_format_reward 的 sample inspection / error analysis，
把它和 sft_lora_small_v2、grpo_lora_small 对比。
```
