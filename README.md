# Evaluation-Driven SFT-DPO-GRPO Reasoning Alignment

本项目基于 Qwen2.5-1.5B/3B-Instruct 构建数学与代码推理对齐系统，参考 Open-R1 推理模型 pipeline，完成评估先行、数据构建、LoRA SFT、DPO、GRPO/RLVR 多阶段后训练实验。

本项目不是单纯的 lm-eval 项目，也不是单纯的微调项目，而是一个：

```text
评估驱动的 SFT-DPO-GRPO/RLVR 数学与代码推理对齐闭环
```

---

## 项目目标

本项目目标是构建一个可逐步扩展的 reasoning alignment 工程流程：

1. 使用 lm-eval 对原始模型进行 baseline 评估；
2. 使用 OpenR1-Math、GSM8K、MATH 构建统一 CoT 数学推理数据；
3. 使用 LoRA / PEFT 进行 SFT；
4. 使用 distilabel-math-preference-dpo 进行 DPO 偏好对齐；
5. 使用 GRPO/RLVR 结合 rule-based reward 优化数学与代码推理；
6. 每个阶段后重新评估原始模型或 LoRA adapter；
7. 保存 lm-eval samples 和自定义 bad cases，进行错误分析；
8. 逐步从 debug 实验扩展到 small / formal 实验。

---

## 当前实验主线

```text
Baseline 评估
→ SFT 数据构造
→ LoRA SFT
→ SFT 后评估
→ DPO 数据构造
→ DPO 训练
→ DPO 后评估
→ GRPO/RLVR 数据构造
→ GRPO/RLVR 训练
→ GRPO 后评估
→ 评估汇总
→ 样本分析
→ 错误类型分析
→ reasoning 错误模式分析
→ targeted small_v2 SFT
→ format-constrained small_v2 SFT
→ small_v2 样本级对比分析
→ prompt-level format eval
→ reward-based format optimization
→ formal 实验扩展
```

目前已经完成：

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
→ prompt-level format eval
→ final-answer reward
→ GRPO format-reward debug
→ GRPO format-reward small run
→ SFT-v2-start GRPO format-reward 384 run
→ exact sample comparison
```

---

## 模型

当前主要使用：

* Qwen/Qwen2.5-1.5B-Instruct

后续可扩展到：

* Qwen/Qwen2.5-3B-Instruct

---

## 数据集

### 数学训练 / SFT 数据

* open-r1/OpenR1-Math-220k
* openai/gsm8k
* EleutherAI/hendrycks_math

### DPO 偏好数据

* argilla/distilabel-math-preference-dpo

### 当前评估任务

* gsm8k_cot / GSM8K

### 后续评估扩展

* MATH / MATH-500
* HumanEval
* MBPP
* EvalPlus

---

## 本地硬件与运行环境

当前本地环境：

```text
CPU：AMD Ryzen 5 7500F
GPU：AMD RX 7800 XT
系统：Windows
开发环境：VS Code + PowerShell + venv
```

由于当前是 Windows + AMD 显卡，本地阶段主要采用 CPU 小规模实验：

```text
device = cpu
dtype = float32
batch_size = 1
limit = 5 / 20
max_steps = 1 / 5 / 10 / 20 / 30
```

当前结果主要用于验证工程链路，不作为正式模型性能。

后续正式训练可迁移到 NVIDIA 云服务器，扩大 batch size、训练步数、数据规模和模型规模。

---

## 项目模块

```text
configs/    训练与评估配置
scripts/    数据构造、训练、评估、汇总脚本
src/        答案抽取、prompt 模板、reward 函数
data/       本地构造的数据文件，默认不提交 Git
outputs/    checkpoint、评估结果、报告，部分报告提交 Git
external/   外部资源，默认不提交 Git
```

核心模块包括：

* lm-eval 标准评估
* 自定义 bad case 分析
* SFT 数据构造
* LoRA SFT
* DPO 偏好对齐
* GRPO/RLVR rule-based reward
* final-answer correctness / format / extractability reward
* reward-based format optimization
* LoRA adapter 评估
* 评估结果汇总
* lm-eval samples 预览
* small 阶段样本错误分析
* small 阶段错误类型汇总
* small 阶段 reasoning 错误模式分析
* targeted SFT small_v2 数据构造、训练与评估
* format-constrained SFT small_v2 数据构造、训练与评估
* prompt-level format eval
* reward-based format optimization
* exact sample comparison
* debug / small 阶段实验报告

---

## 当前进度

### 已完成

* [x] VS Code + PowerShell + venv 本地开发环境
* [x] 项目目录结构搭建
* [x] Git 仓库初始化与阶段性提交
* [x] lm-eval 标准评估接入
* [x] baseline 评估脚本
* [x] LoRA adapter 评估接口
* [x] 自定义 GSM8K bad case 保存脚本
* [x] 数据集读取检查
* [x] SFT debug 数据构造
* [x] LoRA SFT debug 训练
* [x] SFT debug 后评估
* [x] DPO debug 数据构造
* [x] DPO debug 训练
* [x] DPO debug 后评估
* [x] GRPO/RLVR debug 数据构造
* [x] GRPO/RLVR debug 训练
* [x] GRPO/RLVR debug 后评估
* [x] lm-eval 结果汇总脚本
* [x] lm-eval samples 预览脚本
* [x] debug 阶段报告生成
* [x] SFT small 数据构造、训练与评估
* [x] DPO small 数据构造、训练与评估
* [x] GRPO small 数据构造、训练与评估
* [x] small 阶段评估汇总修复，包括 Sample Len、去重、debug / small 说明
* [x] lm-eval samples 阶段识别修复
* [x] small 阶段样本级错误分析
* [x] small 阶段错误类型汇总
* [x] small 阶段 reasoning 错误模式分析
* [x] targeted SFT small_v2 数据构造
* [x] targeted SFT small_v2 训练
* [x] targeted SFT small_v2 后评估
* [x] format-constrained SFT small_v2 数据构造
* [x] format-constrained SFT small_v2 训练
* [x] format-constrained SFT small_v2 后评估
* [x] eval_summary 已加入 sft_lora_small_v2 和 sft_lora_small_v2_format 结果
* [x] lm-eval sample preview 已修复 small_v2 / small_v2_format 阶段识别
* [x] small_v2 样本级对比分析
* [x] prompt-level format eval v1
* [x] prompt-level format eval v2
* [x] final-answer reward 设计与测试
* [x] final-answer reward 接入 GRPOTrainer
* [x] 从 dpo_lora_small 出发的 GRPO format-reward debug / small run / eval
* [x] 从 sft_lora_small_v2 出发的 GRPO format-reward debug
* [x] SFT-v2 reward inspection
* [x] SFT-v2 + max_completion_length=384 的 GRPO format-reward debug
* [x] SFT-v2 + max_completion_length=384 的 GRPO format-reward 5-step small run
* [x] grpo_lora_small_v2_format_reward_384 评估
* [x] SFT-v2 vs GRPO-384 exact sample comparison
* [x] reward-based format optimization 阶段总结

---

## 当前 small 阶段 GSM8K-COT 评估结果

| Stage                                | Sample Len | Flexible Exact Match | Strict Exact Match |
| ------------------------------------ | ---------: | -------------------: | -----------------: |
| baseline                             |          5 |               0.8000 |             0.6000 |
| sft_lora                             |          5 |               0.8000 |             0.4000 |
| dpo_lora                             |          5 |               0.8000 |             0.4000 |
| grpo_lora                            |          5 |               0.8000 |             0.4000 |
| sft_lora_small                       |         20 |               0.4500 |             0.2500 |
| dpo_lora_small                       |         20 |               0.4000 |             0.2000 |
| grpo_lora_small                      |         20 |               0.4000 |             0.2000 |
| sft_lora_small_v2                    |         20 |               0.6000 |             0.2000 |
| sft_lora_small_v2_format             |         20 |               0.3500 |             0.1500 |
| grpo_lora_small_format_reward        |         20 |               0.4000 |             0.2000 |
| grpo_lora_small_v2_format_reward_384 |         20 |               0.6000 |             0.2000 |

注意：由于当前评估样本数很小，并且训练步数较少，上表不能作为正式模型性能，只能作为工程链路和小规模对比记录。

SFT small_v2 使用 targeted 数据补充 `percentage_error`、`money_profit_error`、`unit_rate_error` 后，`flexible-extract` 从 0.4500 提升到 0.6000，但 `strict-match` 从 0.2500 降到 0.2000。
这说明 targeted 数据对答案抽取后的正确率有帮助，但最终输出格式仍需要进一步约束。

format-constrained SFT small_v2 进一步尝试把最终答案强制为 `#### answer` 格式，但结果为 `flexible-extract=0.3500`、`strict-match=0.1500`，相比 targeted small_v2 明显下降。
这说明当前这种直接改写 SFT 文本格式的方案没有改善 strict-match，反而伤害了整体输出质量。

reward-based format optimization 进一步证明：final-answer reward 可以接入 GRPOTrainer，并且从 `sft_lora_small_v2` 出发、将 `max_completion_length` 提升到 384 后可以恢复 reward variance。
但是当前 5-step GRPO small run 只是追平 `sft_lora_small_v2`，没有超过它。

---

## Small 阶段错误分析结论

在完成 SFT small、DPO small、GRPO small 后，项目进一步对 GSM8K-COT small 阶段样本进行了错误分析。

当前错误类型统计如下：

| Stage           | Correct | Strict Format Only Error | Answer Extraction / Format Error | Reasoning / Calc Error |
| --------------- | ------: | -----------------------: | -------------------------------: | ---------------------: |
| sft_lora_small  |       5 |                        4 |                                0 |                     11 |
| dpo_lora_small  |       3 |                        5 |                                1 |                     11 |
| grpo_lora_small |       3 |                        5 |                                1 |                     11 |

其中：

* `correct` 表示 flexible-extract 和 strict-match 都通过；
* `strict_format_only_error` 表示答案数值正确，flexible-extract 通过，但 strict-match 不通过，主要是 strict 输出格式问题；
* `answer_extraction_or_format_error` 表示 lm-eval 的 flexible-extract 判错，但脚本抽取的 pred_answer 与 gold_answer 实际相等，可能是答案抽取或格式兼容问题；
* `reasoning_or_calc_error` 表示 pred_answer 与 gold_answer 不一致，说明答案数值本身错误，更可能是题意理解、推理链或计算错误。

当前 small 阶段的主要问题不是单纯格式问题，而是数学推理 / 计算错误更多。

---

### Reasoning 错误模式进一步分析

进一步对 `reasoning_or_calc_error` 进行关键词规则分析后，发现 small 阶段真正的推理 / 计算错误主要集中在以下几类：

| Stage           | Percentage Error | Money / Profit Error | Unit / Rate Error |
| --------------- | ---------------: | -------------------: | ----------------: |
| sft_lora_small  |                6 |                    4 |                 1 |
| dpo_lora_small  |                6 |                    4 |                 1 |
| grpo_lora_small |                6 |                    4 |                 1 |

当前结果说明：

* `percentage_error` 是最主要错误来源，模型在百分比增长、折扣、比例变化、重启进度等问题上容易误解；
* `money_profit_error` 也较明显，模型在成本、售价、利润、价值变化、总价计算等问题上容易混淆；
* `unit_rate_error` 数量较少，但说明单位、距离、时间、速率关系仍需要加强；
* DPO small 和 GRPO small 没有修复这些错误模式，说明当前 small 设置下的偏好数据和 reward 信号还不足以改善核心数学推理错误。

因此，后续 small_v2 不应该盲目扩大训练步数，而应优先补充：

1. 百分比变化类样本；
2. 金额 / 利润 / 成本类样本；
3. 单位速率类样本；
4. 更明确的最终答案格式约束。

由于当前评估样本数只有 20 条，这些结论只作为 small 阶段诊断依据，不能作为正式性能结论。

---

## Targeted SFT small_v2 实验结论

基于 small 阶段错误模式分析，本项目新增 targeted SFT small_v2 数据构造脚本：

```text
scripts/22_prepare_sft_small_v2_data.py
```

该脚本从 GSM8K 中筛选并构造：

```text
percentage_error: 80
money_profit_error: 80
unit_rate_error: 40
general: 50
```

最终生成：

```text
data/processed/sft_small_v2.jsonl
data/samples/sft_small_v2_preview.jsonl
```

训练配置为：

```text
configs/sft_small_v2.yaml
```

评估配置为：

```text
configs/eval_sft_small_v2_lora.yaml
```

训练后的 LoRA adapter 为：

```text
outputs/checkpoints/sft_lora_small_v2
```

评估结果为：

| Stage             | Sample Len | Flexible Exact Match | Strict Exact Match |
| ----------------- | ---------: | -------------------: | -----------------: |
| sft_lora_small    |         20 |               0.4500 |             0.2500 |
| sft_lora_small_v2 |         20 |               0.6000 |             0.2000 |

阶段性结论：

* targeted SFT small_v2 相比 SFT small，在 `flexible-extract` 上从 0.4500 提升到 0.6000；
* 这说明基于错误模式补充 targeted 数据是有效方向；
* 但 `strict-match` 从 0.2500 降到 0.2000，说明模型最终答案格式仍不稳定；
* 后续应继续做格式约束实验，但不能简单粗暴地改变训练文本格式；
* 当前结论仍然来自 `limit=20` 的小样本评估，不能作为正式性能结论。

---

## Format-constrained SFT small_v2 实验结论

针对 targeted SFT small_v2 的 strict-match 没有提升的问题，本项目进一步新增 format-constrained SFT small_v2 数据构造脚本：

```text
scripts/23_prepare_sft_small_v2_format_data.py
```

该脚本基于：

```text
data/processed/sft_small_v2.jsonl
```

构造了带格式约束的训练数据：

```text
data/processed/sft_small_v2_format.jsonl
data/samples/sft_small_v2_format_preview.jsonl
```

核心格式约束为：

```text
You must put the final answer on the last line in exactly this format:
#### <final_answer>
```

训练配置为：

```text
configs/sft_small_v2_format.yaml
```

评估配置为：

```text
configs/eval_sft_small_v2_format_lora.yaml
```

训练后的 LoRA adapter 为：

```text
outputs/checkpoints/sft_lora_small_v2_format
```

评估结果为：

| Stage                    | Sample Len | Flexible Exact Match | Strict Exact Match |
| ------------------------ | ---------: | -------------------: | -----------------: |
| sft_lora_small_v2        |         20 |               0.6000 |             0.2000 |
| sft_lora_small_v2_format |         20 |               0.3500 |             0.1500 |

阶段性结论：

* format-constrained small_v2 没有提升 strict-match；
* `strict-match` 从 0.2000 下降到 0.1500；
* `flexible-extract` 从 0.6000 下降到 0.3500；
* 这说明当前“直接把 SFT 文本改成强制 `#### answer` 格式”的做法不适合当前 small 设置；
* 更可能的原因是：格式约束文本改变了训练分布，削弱了原本 targeted 数据带来的推理收益；
* 后续如果继续优化格式，应优先尝试更温和的方法，例如只改 prompt、只在 final answer 字段强化格式，或者在 reward / evaluation 层加入格式奖励，而不是整体重写 SFT 文本。

当前实验虽然没有提升指标，但它是有价值的负结果，说明格式优化不能简单依赖强制模板改写，需要更细粒度的格式对齐方案。

---

## Small v2 样本级对比分析结论

为了进一步确认 targeted small_v2 到底修复了哪些题，以及 format-constrained small_v2 到底破坏了哪些题，本项目新增样本级对比脚本：

```text
scripts/24_compare_small_v2_samples.py
```

该脚本对比以下三个阶段在同一批 GSM8K-COT `limit=20` 样本上的表现：

```text
sft_lora_small
sft_lora_small_v2
sft_lora_small_v2_format
```

生成报告：

```text
outputs/reports/small_v2_sample_comparison.csv
outputs/reports/small_v2_sample_comparison.md
```

样本级对比结果如下：

### sft_lora_small → sft_lora_small_v2

| Change Type          | Count |
| -------------------- | ----: |
| fixed_by_targeted_v2 |     3 |
| unchanged            |    17 |

说明：

* targeted small_v2 修复了 3 道 SFT small 原本答错的题；
* 没有出现 `regressed_in_targeted_v2`，说明在当前 20 条样本中 targeted small_v2 没有把原本答对的题变错；
* 被修复的样本包括 unit/rate、money/profit、percentage 相关题型，例如火车距离题、徒步平均速度题、房屋翻修利润题。

### sft_lora_small_v2 → sft_lora_small_v2_format

| Change Type                 | Count |
| --------------------------- | ----: |
| broken_by_format_constraint |     6 |
| fixed_by_format_constraint  |     1 |
| unchanged                   |    13 |

说明：

* format-constrained small_v2 只修复了 1 道 targeted small_v2 原本答错的题；
* 但它破坏了 6 道 targeted small_v2 原本答对的题；
* 被破坏的样本包括简单数量题、百分比题、年薪计算题、单位换算题和平均速度题；
* 这进一步证明直接强制改写 SFT 文本格式会削弱模型原本的推理表现。

样本级结论：

```text
targeted small_v2 是正向实验；
format-constrained small_v2 是负结果实验；
后续应该保留 targeted 数据补强方向，但不要继续采用直接重写 SFT 文本格式的做法。
```

---

## Prompt-level format eval 实验结论

在 targeted small_v2 和 format-constrained small_v2 之后，本项目进一步做了一个低成本格式约束实验：

```text
scripts/25_eval_sft_small_v2_prompt_format.py
```

该实验不重新训练模型，而是继续使用当前最佳 checkpoint：

```text
outputs/checkpoints/sft_lora_small_v2
```

实验方式是：只在评估 prompt 中加入最终答案格式要求，要求模型最后一行输出：

```text
#### <answer>
```

生成报告：

```text
outputs/reports/sft_small_v2_prompt_format_eval.csv
outputs/reports/sft_small_v2_prompt_format_eval.jsonl
outputs/reports/sft_small_v2_prompt_format_eval.md
```

实验结果为：

| Eval Setting              | Flexible Acc | Strict / Format Acc |
| ------------------------- | -----------: | ------------------: |
| lm-eval sft_lora_small_v2 |       0.6000 |              0.2000 |
| prompt-level format eval  |       0.4000 |              0.4000 |
| sft_lora_small_v2_format  |       0.3500 |              0.1500 |

其中 prompt-level format eval 的脚本报告中对应指标为：

```text
flexible_acc = 0.4000
strict_hash_acc = 0.4000
```

阶段性结论：

* prompt-level format eval 将格式命中率提升到 0.4000；
* 但 flexible_acc 从 lm-eval small_v2 的 0.6000 下降到 0.4000；
* 说明只改 prompt 可以增强格式遵循，但会干扰部分数学推理；
* 相比直接训练 `sft_lora_small_v2_format`，prompt-level 方法更好，因为它至少把格式命中率提升到了 0.4000；
* 但它仍然不是最终方案，因为答案正确率下降明显；
* 后续更合理的方向是：在 GRPO/RLVR reward 中加入格式奖励，或者设计更温和的 prompt，而不是直接强制改写全部 SFT 训练文本。

当前实验结论可以概括为：

```text
prompt-level format eval 是中间结果：
它比直接 format-constrained SFT 更好，但仍会损害推理正确率。
```

---

## Prompt-level format eval v2 实验结论

### 实验目的

本实验是对 `sft_lora_small_v2` 的进一步 prompt-level 格式约束测试。

它不重新训练模型，只评估当前最佳 checkpoint：

```text
outputs/checkpoints/sft_lora_small_v2
```

上一版 `prompt-level format eval v1` 使用了较强的格式约束，要求模型输出：

```text
#### <answer>
```

v1 的结果是：

```text
flexible_acc = 0.4000
strict_hash_acc = 0.4000
```

虽然格式命中率提高了，但 flexible accuracy 从原始 `sft_lora_small_v2` 的 `0.6000` 降到了 `0.4000`，说明过强的格式约束会干扰模型推理。

因此本次 v2 改成更温和的格式提示，只要求模型最后一行输出：

```text
Final answer: <answer>
```

目标是测试：能否在减少推理正确率损失的同时，继续保持一定的最终答案格式约束能力。

### 输出文件

```text
outputs/reports/sft_small_v2_prompt_format_v2_eval.csv
outputs/reports/sft_small_v2_prompt_format_v2_eval.jsonl
outputs/reports/sft_small_v2_prompt_format_v2_eval.md
```

### 实验结果

| experiment                  | flexible_acc | format_acc | final_answer_format_hit_rate | strict_hash_acc |
| --------------------------- | -----------: | ---------: | ---------------------------: | --------------: |
| prompt-level format eval v1 |       0.4000 |     0.4000 |                            - |          0.4000 |
| prompt-level format eval v2 |       0.5500 |     0.4000 |                       0.7000 |          0.0000 |

### 结果解释

v2 是一个正向结果。

相比 v1：

```text
flexible_acc: 0.4000 -> 0.5500
format_acc:   0.4000 -> 0.4000
```

这说明更温和的 prompt 格式约束可以减少对推理正确率的伤害，同时保持一定的答案格式收益。

但是 v2 仍然没有完全恢复到原始 `sft_lora_small_v2` 的 flexible score：

```text
原始 sft_lora_small_v2 flexible = 0.6000
prompt format v2 flexible       = 0.5500
```

因此当前结论是：

```text
1. prompt-level format v2 明显优于 v1；
2. 更温和的 prompt 格式控制是有价值的；
3. 但 prompt-level 方法仍然不是最终方案；
4. 后续更合理的方向是 reward-based format optimization。
```

后续进入 reward-based format optimization 时，应该让：

```text
答案正确性 reward 作为主 reward；
最终答案格式 reward 作为辅助 reward；
格式 reward 不能压过答案正确性 reward。
```

也就是说，下一阶段不应该继续强行改 SFT 文本格式，而应该在 GRPO/RLVR 阶段设计更温和的格式奖励。

---

## Reward-based format optimization 阶段结论

### 阶段目标

本阶段在 prompt-level format optimization 之后，进一步实现 reward-based format optimization。

核心目标是：不再只通过 prompt 强制模型输出格式，而是设计 reward，让模型在 GRPO/RLVR 训练中同时优化答案正确性、最终答案格式和答案可抽取性。

### Final-answer reward 设计

本阶段新增并验证了 final-answer reward：

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

也就是说，答案正确性是主 reward，格式只是辅助 reward，避免模型只学会输出 `Final answer: ...` 但答案仍然错误。

相关文件：

```text
src/rewards/final_answer_reward.py
scripts/30_test_final_answer_reward.py
outputs/reports/final_answer_reward_test.md
```

### 从 dpo_lora_small 出发的 GRPO format-reward

首先尝试从：

```text
outputs/checkpoints/dpo_lora_small
```

出发做 GRPO format-reward。

该分支证明：

```text
1. final-answer reward 可以接入 GRPOTrainer；
2. GRPO format-reward 可以训练并保存 adapter；
3. format-reward 专用 prompt 数据可以产生 reward 方差；
4. 但最终评估没有超过当前最佳 SFT small_v2。
```

评估结果：

```text
grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000

sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000
```

因此，从 `dpo_lora_small` 出发的 5-step GRPO format-reward 没有带来最终指标提升。

### 从 sft_lora_small_v2 出发的 GRPO format-reward

由于当前最佳 checkpoint 是：

```text
outputs/checkpoints/sft_lora_small_v2
```

所以继续尝试从 `sft_lora_small_v2` 出发做 GRPO format-reward。

初始 `max_completion_length=256` 的 debug 结果没有 reward 方差：

```text
reward_std = 0
frac_reward_zero_std = 1
grad_norm = 0
```

随后通过 reward inspection 发现，问题主要来自生成长度和截断。将生成长度提高到 384 后，reward variance 恢复：

```text
max_completion_length = 384
reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7084
```

因此继续进行了 5-step small-scale GRPO format-reward 训练：

```text
start_adapter_path: outputs/checkpoints/sft_lora_small_v2
output_dir: outputs/checkpoints/grpo_lora_small_v2_format_reward_384
max_completion_length: 384
num_generations: 4
max_steps: 5
```

训练过程中多数 step 有有效 reward 方差：

```text
step 1 reward_std = 0.5774, grad_norm = 0.7084
step 2 reward_std = 0.5802, grad_norm = 0.7045
step 4 reward_std = 0.55,   grad_norm = 0.6637
step 5 reward_std = 0.05,   grad_norm = 0.9398
```

新 checkpoint 成功生成：

```text
outputs/checkpoints/grpo_lora_small_v2_format_reward_384
```

### 最终评估结果

对 `grpo_lora_small_v2_format_reward_384` 进行 GSM8K-COT limit=20 评估，结果为：

```text
grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000
```

与当前最佳 SFT checkpoint 对比：

```text
sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000

grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000
```

结论是：

```text
GRPO-384 追平了 sft_lora_small_v2，但没有超过。
```

### 样本级对比结论

进一步做 exact sample comparison，精确对比：

```text
sft_lora_small_v2
grpo_lora_small_v2_format_reward_384
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

这说明：

```text
GRPO-384 和 SFT small_v2 不只是总分一样；
它们在当前 20 个 GSM8K-COT 样本上逐题表现完全一致。
```

### 阶段最终结论

本阶段最终结论是：

```text
reward-based format optimization pipeline works,
but current small-scale GRPO format-reward training does not yet improve over the best SFT checkpoint.
```

中文总结：

```text
reward-based format optimization 链路已经跑通，
但当前 small-scale GRPO format-reward 训练还没有超过最佳 SFT checkpoint。
```

更具体地说：

```text
1. final-answer reward 可以正常计算；
2. final-answer reward 可以接入 GRPOTrainer；
3. format-reward 专用 prompt 数据是必要的；
4. num_generations=4 可以产生组内比较；
5. max_completion_length=256 对 sft_lora_small_v2 不够；
6. max_completion_length=384 可以恢复 reward 方差；
7. 从 sft_lora_small_v2 出发优于从 dpo_lora_small 出发；
8. 当前 GRPO-384 最终指标追平 sft_lora_small_v2；
9. 逐题对比显示 GRPO-384 没有新增正确样本；
10. 下一步不应该继续盲目增加 GRPO step。
```

相关报告：

```text
outputs/reports/reward_based_format_optimization_stage_summary.md
outputs/reports/grpo_sft_v2_format_reward_384_eval_report.md
outputs/reports/sft_v2_vs_grpo_384_sample_comparison.md
```

---

## 代码推理分支第一阶段：MBPP safe sample-only baseline

在完成数学推理主线和 reward-based format optimization 阶段后，项目新增代码推理分支，用于将整体方向从“数学推理对齐”扩展为“数学 + 代码推理对齐”。

当前代码推理分支没有直接进入训练，而是先完成最小安全评估闭环：

```text
代码推理工具选择
→ MBPP safe sample-only generation
→ sample 简洁检查
→ 静态错误分析
→ 语义错误分析
→ 代码分支阶段总结
```

### 为什么没有直接使用 lm-eval MBPP pass@1

本阶段首先尝试使用 lm-evaluation-harness 的 `mbpp` task，并设置：

```text
--predict_only
--log_samples
```

目标是只保存模型生成样本，不执行生成代码。

但实际运行时，lm-eval 在加载 MBPP task 阶段触发 Hugging Face `code_eval` 安全门：

```text
The "code_eval" metric executes untrusted model-generated code in Python.
...
set the environment variable HF_ALLOW_CODE_EVAL="1"
```

因此当前阶段没有设置：

```text
HF_ALLOW_CODE_EVAL=1
```

也没有在本机直接执行模型生成代码。

这说明代码推理评估和数学推理评估存在关键区别：

```text
数学评估：
模型生成答案
→ 抽取 final answer
→ 对比标准答案

代码评估：
模型生成 Python 代码
→ 执行生成代码
→ 运行测试用例
→ 计算 pass@1
```

由于模型生成代码属于不可信代码，后续正式 pass@1 / EvalPlus 评估需要 sandbox / Docker / WSL / 隔离环境。

### 当前 safe sample-only 方案

为避免直接执行模型生成代码，本阶段新增自定义安全生成脚本：

```text
scripts/41_generate_mbpp_samples_safe.py
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
不运行 MBPP 测试用例
不计算 pass@1
不设置 HF_ALLOW_CODE_EVAL
```

正式生成命令：

```text
python scripts\41_generate_mbpp_samples_safe.py --limit 5 --device cpu --max-new-tokens 256
```

生成样本：

```text
outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl
```

每条样本均记录：

```json
"safe_generate_only": true,
"executed": false
```

### 当前 MBPP limit=5 结果

当前 MBPP safe sample-only baseline 使用：

```text
model = Qwen/Qwen2.5-1.5B-Instruct
task = MBPP sanitized test split
limit = 5
device = cpu
max_new_tokens = 256
```

样本 task_id：

```text
11
12
14
16
17
```

新增 sample 检查脚本：

```text
scripts/42_inspect_mbpp_samples.py
```

新增静态错误分析脚本：

```text
scripts/43_analyze_mbpp_samples_static.py
```

静态分析报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
outputs/reports/code_mbpp_limit5_static_analysis.jsonl
```

语义错误分析报告：

```text
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
```

代码分支阶段总结报告：

```text
outputs/reports/code_reasoning_branch_stage_summary.md
```

### 静态分析结论

静态分析结果：

```text
样本数：5
语法可解析样本数：5/5
包含函数定义样本数：5/5
函数名匹配样本数：5/5
executed=True 样本数：0/5
static_clean：5/5
```

这说明当前 prompt 和清洗逻辑已经能生成结构较干净的函数代码。

但这不代表代码功能正确。

### 语义错误分析结论

人工语义审查结果：

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

注意：这不是正式 pass@1，只是未执行代码条件下的人工语义分析。

当前代码分支最重要的结论是：

```text
代码格式干净不代表代码功能正确。
```

这和数学推理分支中的经验一致：

```text
只优化格式不能保证推理正确；
必须引入任务语义、边界条件和可验证反馈。
```

### 当前新增文件

当前代码推理分支第一阶段新增配置：

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
```

当前提交：

```text
fe60745 Add MBPP safe code reasoning baseline
```

### 下一阶段代码推理路线

后续代码推理分支建议继续：

```text
1. 设计 sandboxed code execution 方案；
2. 接入 HumanEval safe sample-only generation；
3. 接入 EvalPlus / HumanEval+ / MBPP+；
4. 得到正式 pass@1；
5. 构造代码 SFT 数据；
6. 做小规模代码 SFT；
7. 形成数学 + 代码双任务最终总结。
```

当前不建议直接训练代码模型。

原因是：

```text
还没有正式 pass@1 baseline；
还没有隔离执行环境；
还没有确认错误来自模型能力、prompt、样本清洗还是测试覆盖；
直接训练会让问题来源变模糊。
```
## 当前结论

当前阶段已经证明：

1. baseline、SFT、DPO、GRPO/RLVR 的完整工程链路可以跑通；
2. SFT、DPO、GRPO/RLVR 三阶段训练脚本可以串联；
3. 每个阶段都可以保存 LoRA adapter；
4. 每个 LoRA adapter 都可以通过 lm-eval 重新评估；
5. 评估结果可以统一汇总到 CSV 和 Markdown 报告；
6. debug 实验已经扩展到 small 实验；
7. 项目已经具备样本级错误分析能力；
8. 项目已经能够区分 strict 格式错误、答案抽取 / 格式兼容错误和真正的推理 / 计算错误；
9. 项目已经能对 reasoning 错误进行初步模式归因；
10. targeted SFT small_v2 初步验证了基于错误模式补充数据的有效性；
11. format-constrained small_v2 验证了“直接强制改写 SFT 输出模板”在当前 small 设置下并不有效；
12. small_v2 样本级对比进一步证明 targeted small_v2 修复了 3 道错题，而 format 版本破坏了 6 道原本答对的题；
13. prompt-level format eval 证明只改 prompt 可以提升格式命中率，但会降低推理正确率；
14. prompt-level format eval v2 证明温和的 `Final answer: <answer>` 格式约束优于强制 `#### <answer>`；
15. reward-based format optimization 链路已经跑通；
16. final-answer reward 可以接入 GRPOTrainer，并产生 reward 方差；
17. 从 `sft_lora_small_v2` 出发优于从 `dpo_lora_small` 出发；
18. `max_completion_length=384` 可以恢复 reward variance；
19. `grpo_lora_small_v2_format_reward_384` 追平 `sft_lora_small_v2`，但没有超过；
20. 样本级 exact comparison 显示 GRPO-384 和 SFT-v2 在当前 20 条样本上逐题完全一致；
21. 项目具备继续扩展到 MATH、代码推理和正式实验的基础。

---

## 当前最佳 checkpoint

当前 small 阶段最值得保留的是：

```text
outputs/checkpoints/sft_lora_small_v2
```

原因是：

```text
sft_lora_small_v2 flexible-extract = 0.6000
sft_lora_small_v2 strict-match     = 0.2000
```

`grpo_lora_small_v2_format_reward_384` 也达到了同样指标：

```text
grpo_lora_small_v2_format_reward_384 flexible-extract = 0.6000
grpo_lora_small_v2_format_reward_384 strict-match     = 0.2000
```

但样本级 exact comparison 显示，二者在当前 20 条 GSM8K-COT 样本上逐题表现完全一致：

```text
both_correct = 12
both_wrong = 8
grpo_improved = 0
grpo_regressed = 0
```

因此当前最佳 checkpoint 仍然优先保留：

```text
outputs/checkpoints/sft_lora_small_v2
```

同时将以下 checkpoint 作为 reward-based format optimization 的正向中间结果保留：

```text
outputs/checkpoints/grpo_lora_small_v2_format_reward_384
```

不推荐继续沿着以下 checkpoint 优化：

```text
outputs/checkpoints/sft_lora_small_v2_format
```

原因是：

```text
sft_lora_small_v2_format flexible-extract = 0.3500
sft_lora_small_v2_format strict-match     = 0.1500
```

---

## Git 提交记录

当前主要提交包括：

```text
23dfac0 Add reward-based format optimization stage summary
4a8629e Add SFT v2 versus GRPO 384 sample comparison
7481753 Add SFT v2 GRPO format reward 384 evaluation
cb8d58b Add SFT v2 GRPO format reward 384 small run
5ac6a58 Add SFT v2 GRPO format reward debug 384
c365ef9 Add SFT v2 format reward inspection
3f93aed Add GRPO format reward debug from SFT v2
c95a216 Add exact GRPO format reward sample comparison
7887edf Add GRPO format reward evaluation
5c22624 Add GRPO format reward small run
0f7bf0c Fix GRPO format reward debug v2 report content
82325d0 Add GRPO format reward debug v2
c84aedc Improve GRPO format reward inspection
a2101dc Add GRPO format reward inspection
9dc6c5e Add GRPO format reward debug run
0c6e00d Tighten final answer format reward
bacdb91 Add final answer reward test
4237f9b Add prompt format reward diagnosis
20f268e Add softer prompt-level format eval for small v2
489d892 Add prompt-level format eval for small v2
eb6c80b Add small v2 sample comparison analysis
711d684 Fix sample preview stage detection for small v2
80a60ca Update README with format-constrained small v2 results
c22e262 Add format-constrained SFT small v2 experiment
f359fe9 Update README with targeted small v2 results
1c84b84 Add targeted SFT small v2 experiment
7ec6ec7 Update README with reasoning error patterns
44b936d Add small reasoning error pattern analysis
93bbf34 Update README with small error analysis
28bd702 Add small evaluation error type summary
03e9847 Add small evaluation error analysis
24f2598 Fix lm-eval sample stage detection
ec0c777 Update README with small experiment progress
9ea6b7f Improve evaluation summary report
aa20dee Add small GRPO experiment config
87588a6 Add small DPO experiment config
e8e9ced Add small SFT experiment config
d084c98 Add debug evaluation reports
744c707 Initialize reasoning alignment debug pipeline
```

---

## 重要说明

当前所有 debug / small / small_v2 / reward-based format optimization 结果不能作为正式性能结论，原因包括：

```text
debug 阶段：
max_steps = 1
limit = 5

small 阶段：
max_steps = 10 / 20
limit = 20

small_v2 阶段：
max_steps = 30
limit = 20

small_v2_format 阶段：
max_steps = 30
limit = 20

prompt-level format eval：
limit = 20

reward-based format optimization：
max_steps = 1 / 5
limit = 20

本地运行：
device = cpu
dtype = float32
batch_size = 1
```

当前阶段重点是工程验证、闭环搭建和误差分析，而不是追求正式指标提升。

---

## 后续计划

* [ ] 将 reward-based format optimization 阶段结论同步到最终项目报告
* [ ] 设计更温和的格式约束方案，避免直接重写 SFT 文本导致推理能力下降
* [ ] 尝试只在 prompt / final answer 部分加入格式约束
* [ ] 优化 final-answer reward，使其能提供更强但不过度压制 correctness 的格式信号
* [ ] 继续研究 completion 截断问题，尝试缩短 prompt 或继续增加 max_completion_length
* [ ] 基于 small_v2 样本级对比结果继续扩展 targeted 数据
* [ ] 继续细分 reasoning_or_calc_error，例如百分比错误、单位错误、多步计算错误、题意理解错误
* [ ] 增加 MATH / MATH-500 评估
* [ ] 增加 HumanEval / MBPP / EvalPlus 代码推理评估
* [ ] 增加更大规模 SFT / DPO / GRPO 实验配置
* [ ] 增加正式实验报告
* [ ] 增加样本级错误分析和 bad case 分类
* [ ] 后续在 NVIDIA 云服务器上扩展更大 batch、更多 steps、更大模型

---

## 适合简历表述的阶段性成果

当前项目可以概括为：

```text
基于 Qwen2.5-1.5B-Instruct 构建评估驱动的数学推理对齐实验框架，完成 baseline、LoRA SFT、DPO、GRPO/RLVR 多阶段后训练闭环；接入 lm-evaluation-harness，支持 GSM8K-COT 评估、LoRA adapter 评估、样本输出分析、错误类型统计、reasoning 错误模式归因和结果汇总；在本地 CPU 环境下完成 debug、small、targeted small_v2、format-constrained small_v2、prompt-level format eval 与 reward-based format optimization 多级实验验证；通过 targeted 数据将 GSM8K-COT small 评估的 flexible-extract 从 0.4500 提升到 0.6000；进一步通过样本级对比发现 targeted small_v2 修复了 3 道 small 错题且未造成回退，而直接强制格式改写破坏了 6 道原本答对的题；设计 final-answer correctness/format/extractability 组合 reward 并接入 TRL GRPOTrainer，验证从 SFT/DPO checkpoint 出发的 reward-based format optimization 全流程；发现从最佳 SFT checkpoint 出发并提升 max_completion_length 到 384 可恢复 reward variance，但当前 5-step GRPO 在 GSM8K-COT limit=20 上追平而未超过 SFT baseline，为后续扩展到更大规模 RLVR、MATH、HumanEval、MBPP 和云端训练打下工程基础。
```

