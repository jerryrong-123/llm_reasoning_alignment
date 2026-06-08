# SFT-v2 384 GRPO format-reward 评估报告

## 实验目的

本报告记录新 checkpoint 的 lm-eval 结果：

```text
outputs/checkpoints/grpo_lora_small_v2_format_reward_384
```

该 checkpoint 来自以下训练路线：

```text
sft_lora_small_v2
-> GRPO format reward
-> max_completion_length = 384
-> max_steps = 5
```

本次评估的目标是判断：

```text
从当前最佳 SFT checkpoint 出发做 384 长度 GRPO format-reward small 训练后，
是否能超过原始 sft_lora_small_v2。
```

## 评估配置

本次评估配置为：

```text
config: configs/eval_grpo_sft_v2_format_reward_384_lora.yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
peft_path: outputs/checkpoints/grpo_lora_small_v2_format_reward_384
tasks: gsm8k_cot
limit: 20
batch_size: 1
device: cpu
dtype: float32
apply_chat_template: true
output_name: grpo_lora_small_v2_format_reward_384_qwen25_15b_gsm8k_cot_limit20
```

运行命令：

```powershell
python scripts\01_lmeval_eval.py --config configs\eval_grpo_sft_v2_format_reward_384_lora.yaml
```

评估输出目录：

```text
outputs/eval/grpo_lora_small_v2_format_reward_384_qwen25_15b_gsm8k_cot_limit20
```

## 评估结果

lm-eval 输出：

```text
| Task      | Filter           | n-shot | Metric      | Value  | Stderr |
|-----------|------------------|-------:|-------------|-------:|-------:|
| gsm8k_cot | flexible-extract |      8 | exact_match | 0.6000 | 0.1124 |
| gsm8k_cot | strict-match     |      8 | exact_match | 0.2000 | 0.0918 |
```

因此：

```text
grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000
```

## 和已有结果对比

当前关键结果：

```text
sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000

grpo_lora_small:
flexible = 0.4000
strict   = 0.2000

grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000

grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000
```

## 结果解释

本次结果说明：

```text
1. 从 sft_lora_small_v2 出发的 GRPO format-reward 384 small 训练可以成功评估；
2. 新 checkpoint 的 flexible-extract 达到 0.6000；
3. 新 checkpoint 的 strict-match 为 0.2000；
4. 它追平了当前最佳 sft_lora_small_v2；
5. 但它没有超过 sft_lora_small_v2；
6. 它明显好于之前从 dpo_lora_small 出发的 grpo_lora_small_format_reward。
```

也就是说：

```text
从 dpo_lora_small 出发：
grpo_lora_small_format_reward flexible = 0.4000

从 sft_lora_small_v2 出发，并使用 max_completion_length = 384：
grpo_lora_small_v2_format_reward_384 flexible = 0.6000
```

这说明起点 checkpoint 很重要。

## 当前结论

本次实验是一个正向中间结果：

```text
reward-based format optimization pipeline works;
starting from sft_lora_small_v2 is better than starting from dpo_lora_small;
max_completion_length = 384 helps recover reward variance and final eval performance;
but the current 5-step run still does not surpass sft_lora_small_v2.
```

中文结论：

```text
reward-based format optimization 链路已经跑通；
从 sft_lora_small_v2 出发优于从 dpo_lora_small 出发；
max_completion_length=384 可以恢复 reward 方差和最终评估表现；
但当前 5 step 训练只是追平 sft_lora_small_v2，还没有超过。
```

## 下一步方向

下一步不应该立刻继续加训练步数。

更合理的是先做样本级对比：

```text
1. 对比 sft_lora_small_v2 和 grpo_lora_small_v2_format_reward_384；
2. 找出两者答对/答错是否完全一致；
3. 检查 GRPO 384 是否只是数值追平，还是在样本分布上有变化；
4. 如果样本层面有互补，再考虑继续优化；
5. 如果样本完全一致，则说明当前 GRPO 只是没有明显改变模型行为。
```

推荐下一步：

```text
新增 exact sample comparison，把 sft_lora_small_v2 和 grpo_lora_small_v2_format_reward_384 做逐题对比。
```