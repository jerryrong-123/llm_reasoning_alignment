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
→ small 实验扩展
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
max_steps = 1 / 10 / 20
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
* LoRA adapter 评估
* 评估结果汇总
* lm-eval samples 预览
* small 阶段样本错误分析
* small 阶段错误类型汇总
* small 阶段 reasoning 错误模式分析
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

---

## 当前 small 阶段 GSM8K-COT 评估结果

| Stage           | Sample Len | Flexible Exact Match | Strict Exact Match |
| --------------- | ---------: | -------------------: | -----------------: |
| baseline        |          5 |               0.8000 |             0.6000 |
| sft_lora        |          5 |               0.8000 |             0.4000 |
| dpo_lora        |          5 |               0.8000 |             0.4000 |
| grpo_lora       |          5 |               0.8000 |             0.4000 |
| sft_lora_small  |         20 |               0.4500 |             0.2500 |
| dpo_lora_small  |         20 |               0.4000 |             0.2000 |
| grpo_lora_small |         20 |               0.4000 |             0.2000 |

注意：由于当前评估样本数很小，并且训练步数较少，上表不能作为正式模型性能，只能作为工程链路和小规模对比记录。

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
10. 项目具备继续扩展到 MATH、代码推理和正式实验的基础。

---

## Git 提交记录

当前主要提交包括：

```text
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

当前所有 debug / small 结果不能作为正式性能结论，原因包括：

```text
debug 阶段：
max_steps = 1
limit = 5

small 阶段：
max_steps = 10 / 20
limit = 20

本地运行：
device = cpu
dtype = float32
batch_size = 1
```

当前阶段重点是工程验证、闭环搭建和误差分析，而不是追求指标提升。

---

## 后续计划

* [ ] 基于 reasoning 错误模式构建 small_v2 数据，优先补充百分比变化、金额利润、单位速率类样本
* [ ] 增加格式约束实验，提高 strict-match
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
基于 Qwen2.5-1.5B-Instruct 构建评估驱动的数学推理对齐实验框架，完成 baseline、LoRA SFT、DPO、GRPO/RLVR 多阶段后训练闭环；接入 lm-evaluation-harness，支持 GSM8K-COT 评估、LoRA adapter 评估、样本输出分析、错误类型统计、reasoning 错误模式归因和结果汇总；在本地 CPU 环境下完成 debug 与 small 两级实验验证，为后续扩展到 MATH、HumanEval、MBPP 和更大规模训练打下工程基础。
```
