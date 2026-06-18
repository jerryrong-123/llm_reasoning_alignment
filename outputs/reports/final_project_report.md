# Evaluation-Driven SFT-DPO-GRPO Reasoning Alignment 最终项目报告

## 1. 项目概述

本项目基于 `Qwen/Qwen2.5-1.5B-Instruct` 构建一个评估驱动的数学推理对齐实验框架，目标是复现并实现一个可扩展的 SFT-DPO-GRPO/RLVR 后训练闭环。

本项目不是单纯的模型微调项目，也不是单纯的 lm-eval 评估项目，而是一个完整的 reasoning alignment 工程实验系统：

```text
Baseline 评估
→ 数据构造
→ LoRA SFT
→ SFT 后评估
→ DPO 偏好对齐
→ DPO 后评估
→ GRPO/RLVR reward optimization
→ GRPO 后评估
→ 样本级错误分析
→ 错误模式归因
→ targeted 数据优化
→ format optimization
→ reward-based format optimization
→ 阶段报告与 README 总结
```

当前项目主要围绕 GSM8K-COT 数学推理任务展开，后续可以扩展到 MATH、MATH-500、HumanEval、MBPP、EvalPlus 等数学和代码推理任务。

---

## 2. 项目目标

本项目的核心目标包括：

1. 搭建一个本地可运行的 LLM reasoning alignment 工程框架；
2. 使用 lm-evaluation-harness 对 base model 和 LoRA adapter 进行标准化评估；
3. 使用 GSM8K / OpenR1-Math / preference DPO 数据构造训练样本；
4. 完成 LoRA SFT、DPO、GRPO/RLVR 多阶段训练；
5. 设计 rule-based reward 和 final-answer reward；
6. 对每个阶段的模型输出进行样本级对比；
7. 区分格式错误、抽取错误、推理错误和计算错误；
8. 基于错误模式构造 targeted small_v2 数据；
9. 验证 prompt-level 和 reward-based format optimization；
10. 形成一份可以用于简历、面试和后续云端扩展的完整项目报告。

---

## 3. 本地运行环境

当前本地环境为：

```text
CPU: AMD Ryzen 5 7500F
GPU: AMD RX 7800 XT
系统: Windows
开发环境: VS Code + PowerShell + venv
主要运行设备: CPU
dtype: float32
batch_size: 1
```

由于本地是 Windows + AMD 显卡，当前实验主要采用 CPU 小规模设置：

```text
debug 阶段:
max_steps = 1
limit = 5

small 阶段:
max_steps = 10 / 20
limit = 20

small_v2 阶段:
max_steps = 30
limit = 20

reward-based format optimization:
max_steps = 1 / 5
limit = 20
```

因此，当前所有结果都不能作为正式模型性能结论，而是用于验证工程链路、定位误差模式和筛选后续优化方向。

---

## 4. 项目目录结构

项目主要目录如下：

```text
configs/    训练与评估配置
scripts/    数据构造、训练、评估、汇总、样本分析脚本
src/        答案抽取、prompt 模板、reward 函数
data/       本地构造的数据文件
outputs/    checkpoint、评估结果、报告
external/   外部资源
```

关键模块包括：

```text
lm-eval 标准评估
LoRA adapter 评估
SFT 数据构造
DPO 偏好数据构造
GRPO/RLVR 训练
rule-based reward
final-answer reward
评估结果汇总
lm-eval samples 预览
样本级错误分析
reasoning 错误模式分析
targeted 数据构造
prompt-level format eval
reward-based format optimization
exact sample comparison
阶段报告生成
```

---

## 5. 总体实验路线

当前项目已经完成以下主线：

```text
Baseline
→ SFT debug
→ DPO debug
→ GRPO/RLVR debug
→ SFT small
→ DPO small
→ GRPO small
→ evaluation summary
→ small sample error analysis
→ small error type summary
→ small reasoning error pattern analysis
→ targeted SFT small_v2
→ format-constrained SFT small_v2
→ small_v2 sample comparison analysis
→ prompt-level format eval v1
→ prompt-level format eval v2
→ final-answer reward
→ GRPO format-reward debug
→ GRPO format-reward small run
→ SFT-v2-start GRPO format-reward 384 run
→ exact sample comparison
→ reward-based format optimization 阶段总结
→ README 更新
→ 当前项目状态交接文档
```

---

## 6. Baseline / SFT / DPO / GRPO small 阶段结果

### 6.1 small 阶段 GSM8K-COT 结果

当前 small 阶段关键评估结果如下：

| Stage | Sample Len | Flexible Exact Match | Strict Exact Match |
|---|---:|---:|---:|
| baseline | 5 | 0.8000 | 0.6000 |
| sft_lora | 5 | 0.8000 | 0.4000 |
| dpo_lora | 5 | 0.8000 | 0.4000 |
| grpo_lora | 5 | 0.8000 | 0.4000 |
| sft_lora_small | 20 | 0.4500 | 0.2500 |
| dpo_lora_small | 20 | 0.4000 | 0.2000 |
| grpo_lora_small | 20 | 0.4000 | 0.2000 |
| sft_lora_small_v2 | 20 | 0.6000 | 0.2000 |
| sft_lora_small_v2_format | 20 | 0.3500 | 0.1500 |
| grpo_lora_small_format_reward | 20 | 0.4000 | 0.2000 |
| grpo_lora_small_v2_format_reward_384 | 20 | 0.6000 | 0.2000 |

### 6.2 small 阶段初步结论

small 阶段说明：

1. SFT small、DPO small、GRPO small 的训练和评估链路均可跑通；
2. 每个阶段都可以生成 LoRA adapter；
3. 每个 LoRA adapter 都可以通过 lm-eval 加载并评估；
4. DPO small 和 GRPO small 在当前设置下没有超过 SFT small；
5. 后续不能盲目扩大训练步数，而应该做样本级错误分析。

---

## 7. small 阶段错误分析

### 7.1 错误类型分类

small 阶段错误被分为：

```text
correct
strict_format_only_error
answer_extraction_or_format_error
reasoning_or_calc_error
```

错误类型统计如下：

| Stage | Correct | Strict Format Only Error | Answer Extraction / Format Error | Reasoning / Calc Error |
|---|---:|---:|---:|---:|
| sft_lora_small | 5 | 4 | 0 | 11 |
| dpo_lora_small | 3 | 5 | 1 | 11 |
| grpo_lora_small | 3 | 5 | 1 | 11 |

### 7.2 错误分析结论

当前 small 阶段的主要问题不是单纯格式问题，而是数学推理 / 计算错误更多。

进一步的 reasoning 错误模式分析发现，错误主要集中在：

| Stage | Percentage Error | Money / Profit Error | Unit / Rate Error |
|---|---:|---:|---:|
| sft_lora_small | 6 | 4 | 1 |
| dpo_lora_small | 6 | 4 | 1 |
| grpo_lora_small | 6 | 4 | 1 |

结论：

```text
percentage_error 是最主要错误来源；
money_profit_error 也较明显；
unit_rate_error 数量较少但仍存在；
DPO small 和 GRPO small 没有修复这些错误模式。
```

因此，后续应优先补充：

```text
1. 百分比变化类样本；
2. 金额 / 利润 / 成本类样本；
3. 单位速率类样本；
4. 更明确的最终答案格式约束。
```

---

## 8. targeted SFT small_v2 实验

### 8.1 实验动机

基于 small 阶段错误模式分析，项目新增 targeted SFT small_v2 数据构造，重点补充：

```text
percentage_error
money_profit_error
unit_rate_error
general
```

数据构造脚本：

```text
scripts/22_prepare_sft_small_v2_data.py
```

生成文件：

```text
data/processed/sft_small_v2.jsonl
data/samples/sft_small_v2_preview.jsonl
```

训练配置：

```text
configs/sft_small_v2.yaml
```

评估配置：

```text
configs/eval_sft_small_v2_lora.yaml
```

训练后的 checkpoint：

```text
outputs/checkpoints/sft_lora_small_v2
```

### 8.2 targeted SFT small_v2 结果

| Stage | Sample Len | Flexible Exact Match | Strict Exact Match |
|---|---:|---:|---:|
| sft_lora_small | 20 | 0.4500 | 0.2500 |
| sft_lora_small_v2 | 20 | 0.6000 | 0.2000 |

### 8.3 targeted SFT small_v2 结论

targeted SFT small_v2 是正向实验。

结论：

```text
1. flexible-extract 从 0.4500 提升到 0.6000；
2. 基于错误模式补充 targeted 数据是有效方向；
3. strict-match 从 0.2500 降到 0.2000，说明最终答案格式仍不稳定；
4. 后续应继续探索格式约束，但不能简单粗暴地改写训练文本。
```

样本级对比显示：

```text
targeted small_v2 修复了 3 道 SFT small 原本答错的题；
没有出现 regressed_in_targeted_v2；
说明 targeted small_v2 在当前 20 条样本中没有造成回退。
```

---

## 9. format-constrained SFT small_v2 实验

### 9.1 实验动机

由于 targeted SFT small_v2 的 strict-match 没有提升，因此尝试构造 format-constrained SFT small_v2。

数据构造脚本：

```text
scripts/23_prepare_sft_small_v2_format_data.py
```

核心格式约束：

```text
You must put the final answer on the last line in exactly this format:
#### <final_answer>
```

训练配置：

```text
configs/sft_small_v2_format.yaml
```

评估配置：

```text
configs/eval_sft_small_v2_format_lora.yaml
```

训练后的 checkpoint：

```text
outputs/checkpoints/sft_lora_small_v2_format
```

### 9.2 format-constrained SFT small_v2 结果

| Stage | Sample Len | Flexible Exact Match | Strict Exact Match |
|---|---:|---:|---:|
| sft_lora_small_v2 | 20 | 0.6000 | 0.2000 |
| sft_lora_small_v2_format | 20 | 0.3500 | 0.1500 |

### 9.3 format-constrained SFT small_v2 结论

format-constrained SFT small_v2 是负结果实验。

结论：

```text
1. strict-match 从 0.2000 下降到 0.1500；
2. flexible-extract 从 0.6000 下降到 0.3500；
3. 直接强制改写 SFT 训练文本格式会削弱原本 targeted 数据带来的推理收益；
4. 后续不应继续采用整体重写 SFT 文本格式的方法。
```

样本级对比进一步显示：

```text
format-constrained small_v2 只修复 1 道 targeted small_v2 原本答错的题；
但破坏了 6 道 targeted small_v2 原本答对的题。
```

---

## 10. prompt-level format eval 实验

### 10.1 prompt-level format eval v1

v1 不重新训练模型，只在评估 prompt 中加入格式要求：

```text
#### <answer>
```

使用 checkpoint：

```text
outputs/checkpoints/sft_lora_small_v2
```

结果：

```text
flexible_acc = 0.4000
strict_hash_acc = 0.4000
```

结论：

```text
格式命中率提升，但 flexible accuracy 从 0.6000 降到 0.4000；
强格式约束会干扰数学推理。
```

### 10.2 prompt-level format eval v2

v2 使用更温和的格式提示：

```text
Final answer: <answer>
```

结果：

```text
flexible_acc = 0.5500
format_acc = 0.4000
final_answer_format_hit_rate = 0.7000
strict_hash_acc = 0.0000
```

结论：

```text
1. v2 明显优于 v1；
2. 温和格式提示能减少对推理正确率的伤害；
3. 但 v2 仍未恢复到原始 sft_lora_small_v2 的 flexible=0.6000；
4. prompt-level format optimization 仍不是最终方案；
5. 后续更合理方向是 reward-based format optimization。
```

---

## 11. reward-based format optimization

### 11.1 实验动机

prompt-level format eval 说明：

```text
只改 prompt 可以提升格式遵循；
但可能干扰推理正确率；
直接改写 SFT 文本格式更会破坏模型表现。
```

因此，本项目进一步设计 reward-based format optimization：

```text
在 GRPO/RLVR 阶段加入 final-answer reward；
让答案正确性作为主 reward；
让最终答案格式作为辅助 reward。
```

### 11.2 final-answer reward 设计

reward 模块：

```text
src/rewards/final_answer_reward.py
```

reward 结构：

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

这样可以避免模型只学会输出 `Final answer: ...`，但答案仍然错误。

### 11.3 final-answer reward 测试结果

测试结果：

```text
correctness_count = 11
correctness_rate = 0.5500
format_hit_count = 7
format_hit_rate = 0.3500
extractable_count = 20
extractable_rate = 1.0000
avg_total_reward = 0.6850
```

结论：

```text
final-answer reward 可以正常计算；
correctness_reward 保持主导地位；
format_reward 不会压过答案正确性；
可以进入 GRPO 接入测试。
```

---

## 12. 从 dpo_lora_small 出发的 GRPO format-reward

### 12.1 debug v1

从 `dpo_lora_small` 出发，初始 debug 结果：

```text
reward_mean = 0.1
reward_std = 0
frac_reward_zero_std = 1
loss = 0
grad_norm = 0
adapter_config.json = True
```

结论：

```text
reward 成功接入 GRPOTrainer；
adapter 可以保存；
但 reward_std=0，没有有效学习信号。
```

### 12.2 reward inspection 与 debug v2

通过 reward inspection 发现，更明确的 format prompt 和更长生成长度可以产生 reward 方差。

debug v2 结果：

```text
reward_mean = 0.6
reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7355
adapter_config.json = True
```

结论：

```text
format-reward 专用 prompt 数据是必要的；
num_generations=4 比 num_generations=2 更适合；
max_completion_length=256 能产生有效 reward 方差。
```

### 12.3 dpo-start small run 结果

5 step 训练结果中，多数 step 有 reward 方差：

```text
step 1 reward_std = 0.5774, grad_norm = 0.7355
step 2 reward_std = 0.5354, grad_norm = 1.584
step 3 reward_std = 0,      grad_norm = 0
step 4 reward_std = 0.5,    grad_norm = 2.829
step 5 reward_std = 0.05,   grad_norm = 0.788
```

评估结果：

```text
grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000
```

对比最佳 SFT：

```text
sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000
```

结论：

```text
从 dpo_lora_small 出发的 GRPO format-reward 训练链路成功；
但最终指标没有超过 sft_lora_small_v2。
```

---

## 13. 从 sft_lora_small_v2 出发的 GRPO format-reward

### 13.1 256 debug

因为当前最佳 checkpoint 是：

```text
outputs/checkpoints/sft_lora_small_v2
```

所以继续测试从 SFT-v2 出发做 GRPO format-reward。

初始 `max_completion_length=256` debug 结果：

```text
reward_mean = 0.1
reward_std = 0
frac_reward_zero_std = 1
loss = 0
grad_norm = 0
adapter_config.json = True
```

结论：

```text
从 sft_lora_small_v2 出发的训练链路可以跑通；
但 max_completion_length=256 时没有有效 reward 方差。
```

### 13.2 SFT-v2 reward inspection

inspection 使用更长生成长度后发现：

```text
total_generations = 12
correctness_count = 6
correctness_rate = 0.5000
format_hit_count = 5
format_hit_rate = 0.4167
extractable_count = 12
extractable_rate = 1.0000
avg_total_reward = 0.6417
unique_reward_values = [0.1, 0.2, 1.1, 1.2]
```

结论：

```text
sft_lora_small_v2 不是不能产生 reward 方差；
问题主要来自生成长度和截断；
当 max_new_tokens=384 时，可以观察到不同 reward。
```

### 13.3 384 debug

将 `max_completion_length` 提升到 384 后，debug 结果：

```text
reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7084
adapter_config.json = True
```

结论：

```text
max_completion_length=384 可以让从 sft_lora_small_v2 出发的 GRPO debug 产生有效 reward 方差。
```

### 13.4 384 small run

5 step small run 配置：

```text
start_adapter_path: outputs/checkpoints/sft_lora_small_v2
output_dir: outputs/checkpoints/grpo_lora_small_v2_format_reward_384
max_completion_length: 384
num_generations: 4
max_steps: 5
```

训练结果：

```text
step 1 reward_mean = 0.600, reward_std = 0.5774, grad_norm = 0.7084
step 2 reward_mean = 0.650, reward_std = 0.5802, grad_norm = 0.7045
step 3 reward_mean = 0.100, reward_std = 0,      grad_norm = 0
step 4 reward_mean = 0.375, reward_std = 0.55,   grad_norm = 0.6637
step 5 reward_mean = 0.125, reward_std = 0.05,   grad_norm = 0.9398
```

新 checkpoint：

```text
outputs/checkpoints/grpo_lora_small_v2_format_reward_384
```

结论：

```text
从 sft_lora_small_v2 出发；
使用 max_completion_length=384；
进行 5 step GRPO format-reward；
训练链路成功，并且多数 step 有有效 reward 方差。
```

---

## 14. grpo_lora_small_v2_format_reward_384 评估

评估配置：

```text
configs/eval_grpo_sft_v2_format_reward_384_lora.yaml
```

评估结果：

```text
grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000
```

对比：

```text
sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000

grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000

grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000
```

结论：

```text
从 sft_lora_small_v2 出发明显优于从 dpo_lora_small 出发；
max_completion_length=384 可以恢复最终评估表现；
但当前 5-step GRPO 仍然只是追平 sft_lora_small_v2，没有超过。
```

---

## 15. SFT-v2 vs GRPO-384 样本级对比

精确读取的 sample 文件：

```text
sft_lora_small_v2:
outputs/eval/sft_lora_small_v2_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__sft_lora_small_v2/samples_gsm8k_cot_2026-06-07T19-15-00.209160.jsonl

grpo_lora_small_v2_format_reward_384:
outputs/eval/grpo_lora_small_v2_format_reward_384_qwen25_15b_gsm8k_cot_limit20/outputs__checkpoints__grpo_lora_small_v2_format_reward_384/samples_gsm8k_cot_2026-06-08T21-44-10.128064.jsonl
```

样本级结果：

```text
sft_lora_small_v2: 12/20 = 0.6000
grpo_lora_small_v2_format_reward_384: 12/20 = 0.6000
```

分类结果：

```text
both_correct = 12
both_wrong = 8
grpo_improved = 0
grpo_regressed = 0
```

结论：

```text
GRPO-384 和 SFT small_v2 不只是总分一样；
它们在当前 20 个 GSM8K-COT 样本上逐题表现完全一致。
```

这说明当前 GRPO-384 没有新增正确样本，也没有造成回退，只是复现了 SFT-v2 的样本级表现。

---

## 16. 当前最佳 checkpoint

当前最值得保留的是：

```text
outputs/checkpoints/sft_lora_small_v2
```

原因：

```text
sft_lora_small_v2 flexible-extract = 0.6000
sft_lora_small_v2 strict-match     = 0.2000
```

`grpo_lora_small_v2_format_reward_384` 也达到同样指标：

```text
grpo_lora_small_v2_format_reward_384 flexible-extract = 0.6000
grpo_lora_small_v2_format_reward_384 strict-match     = 0.2000
```

但 exact sample comparison 显示：

```text
both_correct = 12
both_wrong = 8
grpo_improved = 0
grpo_regressed = 0
```

因此当前实际最佳 checkpoint 仍然优先保留：

```text
outputs/checkpoints/sft_lora_small_v2
```

`outputs/checkpoints/grpo_lora_small_v2_format_reward_384` 作为 reward-based format optimization 链路跑通的正向中间结果保留。

不推荐继续沿着以下 checkpoint 优化：

```text
outputs/checkpoints/sft_lora_small_v2_format
```

因为它的结果为：

```text
flexible-extract = 0.3500
strict-match     = 0.1500
```

---

## 17. 当前阶段总论

当前项目已经证明：

```text
1. baseline、SFT、DPO、GRPO/RLVR 多阶段链路可以跑通；
2. LoRA adapter 可以训练、保存、加载和评估；
3. lm-eval 可以用于标准化 GSM8K-COT 评估；
4. 项目具备样本级错误分析能力；
5. targeted 数据补强可以提升 flexible-extract；
6. 直接强制 SFT 格式会破坏推理；
7. 温和 prompt-level format eval 更好但仍不是最终方案；
8. final-answer reward 可以接入 GRPOTrainer；
9. 从最佳 SFT checkpoint 出发做 GRPO 更合理；
10. max_completion_length=384 可以恢复 reward variance；
11. 当前 GRPO-384 追平但没有超过 SFT-v2；
12. 项目已经具备迁移到云服务器扩大实验的基础。
```

reward-based format optimization 的最终结论是：

```text
reward-based format optimization pipeline works,
but current small-scale GRPO format-reward training does not yet improve over the best SFT checkpoint.
```

中文总结：

```text
reward-based format optimization 链路已经跑通，
但当前 small-scale GRPO format-reward 训练还没有超过最佳 SFT checkpoint。
```

---

## 18. 当前项目价值

本项目的价值不只在于某一个指标是否提升，而在于完整实现了一个 evaluation-driven reasoning alignment pipeline。

当前项目已经具备以下能力：

```text
1. 能完成 baseline / SFT / DPO / GRPO 多阶段训练；
2. 能统一评估 base model 和 LoRA adapter；
3. 能保存并分析 lm-eval sample；
4. 能做错误类型分类；
5. 能根据错误模式构造 targeted 数据；
6. 能验证正向实验和负结果实验；
7. 能设计并接入 reward；
8. 能进行 reward inspection；
9. 能定位 reward_std=0 的原因；
10. 能做 exact sample comparison；
11. 能把阶段实验写入 README 和报告。
```

这说明项目已经从“跑通微调”升级为“评估驱动的对齐实验系统”。

---

## 19. 当前限制

当前限制主要包括：

```text
1. 评估 limit=20，样本量太小；
2. 本地 CPU 训练速度慢；
3. max_steps 较少，不能代表正式训练效果；
4. 没有在大规模数据上验证；
5. 没有在 MATH / MATH-500 上验证；
6. 代码推理分支还没有正式加入；
7. 当前 GRPO reward 仍然比较简单；
8. completions/clipped_ratio 多次为 1，说明生成截断问题仍然存在。
```

因此，当前结论只能作为 small-scale 工程验证结论，不能作为正式模型性能结论。

---

## 20. 后续计划

### 20.1 最优先方向：整理项目最终材料

当前最推荐的下一步是整理项目最终材料，包括：

```text
1. 最终项目报告；
2. 简历项目描述；
3. 面试讲解稿；
4. 项目结构说明；
5. 实验结果表；
6. 正向结果和负结果总结；
7. 后续扩展计划。
```

当前这个报告可以作为最终材料的主文档。

### 20.2 云服务器扩展

如果继续提升指标，建议迁移到 NVIDIA 云服务器。

云服务器阶段可以扩大：

```text
batch size
max_steps
训练数据量
评估 limit
模型规模
```

迁移前需要准备：

```text
requirements.txt
运行命令清单
checkpoint / data / outputs 迁移清单
GPU 配置建议
```

### 20.3 代码推理分支

如果要增强简历项目完整度，可以加入代码推理分支：

```text
HumanEval
MBPP
EvalPlus
```

需要补充：

```text
代码推理 baseline eval
代码推理 SFT 数据构造
代码推理错误分析
代码推理 reward / eval pipeline
```

### 20.4 reward 优化方向

如果继续优化 reward-based format optimization，可以尝试：

```text
1. 更细粒度的 format reward；
2. 对过长输出加入轻微惩罚；
3. 缩短 prompt，减少 completion 截断；
4. 增加 max_completion_length；
5. 增加 num_generations；
6. 增大训练数据量；
7. 从 sft_lora_small_v2 出发进行更长训练；
8. 在更大样本评估集上确认是否真实提升。
```

---

## 21. 当前可写入简历的表述

当前项目可以写成：

```text
基于 Qwen2.5-1.5B-Instruct 构建评估驱动的数学推理对齐实验框架，完成 baseline、LoRA SFT、DPO、GRPO/RLVR 多阶段后训练闭环；接入 lm-evaluation-harness，支持 GSM8K-COT 评估、LoRA adapter 评估、样本输出分析、错误类型统计、reasoning 错误模式归因和结果汇总；在本地 CPU 环境下完成 debug、small、targeted small_v2、format-constrained small_v2、prompt-level format eval 与 reward-based format optimization 多级实验验证；通过 targeted 数据将 GSM8K-COT small 评估的 flexible-extract 从 0.4500 提升到 0.6000；进一步设计 final-answer correctness/format/extractability 组合 reward 并接入 TRL GRPOTrainer，验证从 SFT/DPO checkpoint 出发的 reward-based format optimization 全流程；发现从最佳 SFT checkpoint 出发并提升 max_completion_length 到 384 可恢复 reward variance，但当前 5-step GRPO 在 GSM8K-COT limit=20 上追平而未超过 SFT baseline，为后续扩展到更大规模 RLVR、MATH、HumanEval、MBPP 和云端训练打下工程基础。
```

---

## 22. 最终结论

本项目当前已经完成一个完整的 evaluation-driven SFT-DPO-GRPO reasoning alignment debug / small-scale pipeline。

当前最重要结论是：

```text
1. targeted 数据补强是有效方向；
2. 直接强制格式 SFT 是负结果；
3. prompt-level format eval 有帮助但会干扰推理；
4. reward-based format optimization 链路已经跑通；
5. 从最佳 SFT checkpoint 出发优于从 DPO checkpoint 出发；
6. max_completion_length=384 对恢复 reward variance 很关键；
7. 当前 GRPO-384 追平但没有超过 SFT-v2；
8. 项目已经具备继续上云扩大实验、加入代码推理分支和整理最终展示材料的基础。
```
## 22. 代码推理分支第一阶段：MBPP safe sample-only baseline

在完成数学推理主线和 reward-based format optimization 阶段后，本项目进一步加入代码推理分支，用于把项目从单一数学推理对齐扩展为“数学 + 代码推理对齐”。

当前代码推理分支没有直接进入代码 SFT 或 GRPO，而是先建立最小安全评估闭环：

```text
代码推理工具选择
→ MBPP safe sample-only generation
→ sample 简洁检查
→ 静态错误分析
→ 语义错误分析
→ 代码分支阶段总结
```

### 22.1 代码评估与数学评估的差异

数学推理评估通常是：

```text
模型生成答案
→ 抽取 final answer
→ 与标准答案比较
```

而代码推理评估通常是：

```text
模型生成 Python 代码
→ 执行生成代码
→ 运行测试用例
→ 计算 pass@1
```

因此代码评估多了一个关键安全问题：模型生成代码属于不可信代码，不能在本机无隔离地直接执行。

本项目当前阶段没有设置：

```text
HF_ALLOW_CODE_EVAL=1
```

也没有执行模型生成代码。

### 22.2 lm-eval MBPP predict_only 的安全门发现

本阶段首先尝试使用 lm-evaluation-harness 的 `mbpp` 任务，并设置：

```text
--predict_only
--log_samples
```

目标是只保存模型输出，不执行生成代码。

但实际运行时，lm-eval 在加载 MBPP task 阶段触发 Hugging Face `code_eval` metric 的安全门：

```text
The "code_eval" metric executes untrusted model-generated code in Python.
...
set the environment variable HF_ALLOW_CODE_EVAL="1"
```

因此当前阶段没有强行设置 `HF_ALLOW_CODE_EVAL=1`，而是改为自定义 safe sample-only generation。

这一点说明代码推理分支不能简单照搬数学推理评估链路，需要把“生成”和“执行评估”拆成两个阶段：

```text
第一阶段：只生成代码样本，不执行代码；
第二阶段：在 sandbox / Docker / WSL / EvalPlus 环境中执行测试。
```

### 22.3 MBPP safe sample-only baseline

本阶段新增安全生成脚本：

```text
scripts/41_generate_mbpp_samples_safe.py
```

运行命令为：

```text
python scripts\41_generate_mbpp_samples_safe.py --limit 5 --device cpu --max-new-tokens 256
```

该脚本只做：

```text
加载 MBPP sanitized test split
构造代码生成 prompt
调用 Qwen/Qwen2.5-1.5B-Instruct 生成代码
抽取 Python 函数
保存 raw_prediction 和 extracted_code
```

明确不做：

```text
不执行模型生成代码
不运行测试用例
不计算 pass@1
不设置 HF_ALLOW_CODE_EVAL
```

输出样本路径为：

```text
outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl
```

每条样本均标记：

```json
"safe_generate_only": true,
"executed": false
```

当前 5 条样本 task_id 为：

```text
11
12
14
16
17
```

### 22.4 静态错误分析

本阶段新增静态错误分析脚本：

```text
scripts/43_analyze_mbpp_samples_static.py
```

输出报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
outputs/reports/code_mbpp_limit5_static_analysis.jsonl
```

静态分析检查内容包括：

```text
是否为空
是否残留 markdown fence
是否包含 print / assert / 测试代码
是否能通过 ast.parse
是否包含函数定义
函数名是否匹配测试用例
是否可能被截断
是否 executed=false
```

静态分析结果为：

```text
样本数：5
语法可解析样本数：5/5
包含函数定义样本数：5/5
函数名匹配样本数：5/5
executed=True 样本数：0/5
static_clean = 5/5
```

这说明当前 prompt 和输出清洗逻辑已经可以得到结构上较干净的函数代码。

但 static_clean 只能说明格式和语法层面干净，不能证明功能正确。

### 22.5 语义错误分析

本阶段进一步完成人工语义错误分析：

```text
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
```

人工语义审查结果为：

| task_id | 静态结果 | 语义判断 | 主要问题 |
|---:|---|---|---|
| 11 | static_clean | 可疑 / 可能错误 | 删除前两个 occurrence，而不是 first + last |
| 12 | static_clean | 大概率正确 | 暂无明显问题 |
| 14 | static_clean | 大概率错误 | 三棱柱体积少了 1/2 |
| 16 | static_clean | 大概率错误 | 条件过宽，未严格建模下划线连接模式 |
| 17 | static_clean | 大概率正确 | 暂无明显问题 |

汇总：

```text
static_clean = 5/5
manual semantic likely correct = 2/5
manual semantic suspicious/wrong = 3/5
executed = 0/5
```

这里的 `manual semantic likely correct = 2/5` 不是正式 pass@1，而是在不执行模型生成代码前提下的人工语义判断。

当前代码推理 baseline 暴露出的主要问题不是格式问题，而是语义问题：

```text
公式少系数
字符串规则过宽
根据 visible tests 写 shortcut
缺少 hidden/general cases 泛化
```

### 22.6 与数学推理主线的关系

代码推理分支和数学推理分支得到了一致的工程经验：

```text
只优化格式不能保证推理正确。
```

在数学推理主线中，format-constrained SFT small_v2 虽然强化了输出格式，但 flexible accuracy 下降，说明强制格式可能破坏推理能力。

在代码推理分支中，5 条 MBPP 样本全部 `static_clean`，但人工语义分析显示只有 2/5 大概率正确，说明代码格式、语法和函数名正确也不能代表功能正确。

因此，项目后续的核心方向不是继续单纯压格式，而是引入：

```text
任务语义
边界条件
测试反馈
execution-based reward
sandboxed evaluation
```

### 22.7 当前新增文件与提交

代码推理分支第一阶段新增配置：

```text
configs/eval_code_baseline_mbpp.yaml
```

新增脚本：

```text
scripts/40_run_code_mbpp_baseline_safe.py
scripts/41_generate_mbpp_samples_safe.py
scripts/42_inspect_mbpp_samples.py
scripts/43_analyze_mbpp_samples_static.py
```

新增报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
outputs/reports/code_mbpp_limit5_static_analysis.jsonl
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
outputs/reports/code_reasoning_branch_stage_summary.md
outputs/reports/readme_code_reasoning_branch_section.md
```

相关 Git 提交：

```text
fe60745 Add MBPP safe code reasoning baseline
498d40f Update README with MBPP code reasoning branch
```

### 22.8 当前阶段结论

代码推理分支第一阶段已经完成最小安全闭环：

```text
1. 确认 lm-eval MBPP predict_only 会触发 code_eval 安全门；
2. 没有设置 HF_ALLOW_CODE_EVAL；
3. 改用自定义 safe sample-only generation；
4. 成功生成 MBPP limit=5 样本；
5. 确认所有样本 executed=false；
6. 完成 sample 简洁检查；
7. 完成静态错误分析；
8. 完成人工语义错误分析；
9. 将代码推理分支写入 README。
```

当前还没有正式 pass@1，因为没有执行模型生成代码，也没有接入 sandbox / EvalPlus。

下一阶段建议继续：

```text
1. 设计 sandboxed code execution 方案；
2. 接入 HumanEval safe sample-only generation；
3. 接入 EvalPlus / HumanEval+ / MBPP+；
4. 得到正式 pass@1；
5. 构造代码 SFT 数据；
6. 做小规模代码 SFT；
7. 形成数学 + 代码双任务最终总结。
```

# HumanEval safe sample-only stage summary

## 1. 阶段背景

在完成 MBPP safe sample-only baseline 和代码执行沙箱方案文档后，本阶段继续把代码推理分支扩展到 HumanEval。

当前阶段仍然遵守代码执行安全边界：

不设置 HF_ALLOW_CODE_EVAL=1。

不执行模型生成代码。

不运行 HumanEval unit tests。

不运行 EvalPlus tests。

不计算正式 pass@1。

本阶段目标不是得到正式代码评估分数，而是先完成 HumanEval 的安全样本生成、样本检查、静态分析和人工语义分析。

## 2. 本阶段新增内容

本阶段新增配置文件：

configs/eval_code_baseline_humaneval.yaml

本阶段新增脚本：

scripts/44_generate_humaneval_samples_safe.py

scripts/45_inspect_humaneval_samples.py

scripts/46_analyze_humaneval_samples_static.py

本阶段新增报告：

outputs/reports/code_humaneval_limit5_static_analysis.md

outputs/reports/code_humaneval_limit5_static_analysis.jsonl

outputs/reports/code_humaneval_limit5_semantic_error_analysis.md

## 3. HumanEval 数据集加载结果

本阶段成功加载 HumanEval 数据集：

dataset = openai/openai_humaneval

split = test

num_rows = 164

字段包括：

task_id

prompt

canonical_solution

test

entry_point

这说明 HumanEval 数据集接入成功，后续可以继续扩展到更大 limit 或 EvalPlus。

## 4. HumanEval safe sample-only generation

本阶段先运行 limit=1 验证链路：

python scripts\44_generate_humaneval_samples_safe.py --limit 1 --device cpu --max-new-tokens 256

limit=1 生成成功：

task_id = HumanEval/0

entry_point = has_close_elements

safe_generate_only = True

executed = False

随后运行 limit=5：

python scripts\44_generate_humaneval_samples_safe.py --limit 5 --device cpu --max-new-tokens 256

limit=5 生成成功，样本包括：

HumanEval/0 has_close_elements

HumanEval/1 separate_paren_groups

HumanEval/2 truncate_number

HumanEval/3 below_zero

HumanEval/4 mean_absolute_deviation

输出文件：

outputs/eval/code_baseline_qwen25_15b_humaneval_limit5_safe_samples/samples_humaneval_safe_generate_only.jsonl

每条样本均标记：

safe_generate_only = true

executed = false

## 5. HumanEval 样本检查结果

本阶段新增样本检查脚本：

scripts/45_inspect_humaneval_samples.py

该脚本只读取样本，不执行模型生成代码。

检查结果确认：

样本数 = 5

所有样本 task_id 正常

所有样本 entry_point 正常

所有样本 safe_generate_only = True

所有样本 executed = False

所有样本 extracted_code 存在

## 6. HumanEval 静态分析结果

本阶段新增静态分析脚本：

scripts/46_analyze_humaneval_samples_static.py

该脚本只做静态分析，不执行模型生成代码。

静态分析报告：

outputs/reports/code_humaneval_limit5_static_analysis.md

outputs/reports/code_humaneval_limit5_static_analysis.jsonl

静态分析结果：

样本数：5

语法可解析样本数：5/5

函数名匹配样本数：5/5

executed=False 样本数：5/5

static_clean：5/5

这说明 HumanEval limit=5 的生成代码在语法、函数名和结构层面比较干净。

但 static_clean 不能代表功能正确，也不能等同于 pass@1。

## 7. HumanEval/3 缩进核对

在样本预览阶段，HumanEval/3 below_zero 的终端显示看起来像存在缩进问题。

因此本阶段额外核对了 JSONL 静态分析结果。

核对结果显示：

task_id = HumanEval/3

entry_point = below_zero

syntax_ok = True

function_names = below_zero

function_name_match = True

issues = static_clean

真实 extracted_code 为：

def below_zero(operations: List[int]) -> bool:
balance = 0

```
for op in operations:
    balance += op

    if balance < 0:
        return True

return False
```

因此第 152 步看到的缩进异常只是终端预览显示问题，不是真实代码问题。

## 8. HumanEval 人工语义分析结果

本阶段新增人工语义分析报告：

outputs/reports/code_humaneval_limit5_semantic_error_analysis.md

人工语义判断结果：

static_clean = 5/5

manual semantic likely correct = 4/5

manual semantic suspicious/wrong = 1/5

executed = 0/5

formal pass@1 = not evaluated

逐题结果：

HumanEval/0 has_close_elements：likely correct

HumanEval/1 separate_paren_groups：likely wrong

HumanEval/2 truncate_number：likely correct

HumanEval/3 below_zero：likely correct

HumanEval/4 mean_absolute_deviation：likely correct

## 9. 当前主要错误样本

当前 HumanEval limit=5 中最明显的问题是：

HumanEval/1 separate_paren_groups

该题要求把多个平衡括号组拆分出来。

模型生成代码在每次遇到右括号时执行：

result.append("".join(stack))

但这会把当前栈内容加入结果，而不是把完整括号组加入结果。

因此该代码虽然语法正确、函数名匹配、static_clean，但语义逻辑不符合题意。

这再次说明：

static_clean 不等于 functional correctness。

## 10. 与 MBPP 分支的对比

MBPP limit=5 阶段结果：

static_clean = 5/5

manual semantic likely correct = 2/5

manual semantic suspicious/wrong = 3/5

executed = 0/5

HumanEval limit=5 阶段结果：

static_clean = 5/5

manual semantic likely correct = 4/5

manual semantic suspicious/wrong = 1/5

executed = 0/5

两者共同说明：

模型可以生成格式和语法较干净的代码。

函数名匹配不代表语义正确。

静态分析不能代替执行测试。

正式代码能力仍然需要 sandboxed execution 或 EvalPlus pass@1 验证。

## 11. 阶段结论

HumanEval safe sample-only baseline 已经跑通。

当前阶段完成了：

HumanEval 数据集加载。

HumanEval limit=1 生成验证。

HumanEval limit=5 safe sample-only generation。

HumanEval 样本检查。

HumanEval 静态分析。

HumanEval 人工语义分析。

HumanEval baseline Git 提交。

当前最新相关提交：

d12681f Add HumanEval safe code reasoning baseline

## 12. 后续计划

下一步应该把 HumanEval 阶段总结写入 README 和 final_project_report。

之后再进入：

EvalPlus / HumanEval+ / MBPP+ 接入调研。

sandboxed execution 原型。

小样本正式 pass@1。

根据错误样本构造 code SFT 数据。

小规模 code SFT。

数学 + 代码双任务最终总结。

当前仍然不建议直接开启 HF_ALLOW_CODE_EVAL=1。

原因是：

正式代码评估必须先隔离执行环境，否则模型生成代码可能带来安全风险。

