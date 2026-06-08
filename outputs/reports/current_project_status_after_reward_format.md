# 当前项目状态交接文档：reward-based format optimization 之后

## 1. 当前项目阶段

当前项目已经完成到：

```text
Baseline
→ SFT debug
→ DPO debug
→ GRPO/RLVR debug
→ SFT small
→ DPO small
→ GRPO small
→ small 阶段评估汇总
→ small 阶段错误分析
→ reasoning 错误模式分析
→ targeted SFT small_v2
→ format-constrained SFT small_v2
→ small_v2 样本级对比
→ prompt-level format eval v1
→ prompt-level format eval v2
→ final-answer reward 设计
→ reward-based format optimization
→ SFT-v2-start GRPO format reward 384
→ exact sample comparison
→ README 阶段总结更新
```

当前 Git 最新提交为：

```text
cbde18e Update README with reward-based format optimization results
```

当前 Git 工作区应为干净状态：

```text
git status --short
```

预期无输出。

---

## 2. 当前最佳 checkpoint

当前 small 阶段最值得保留的 checkpoint 是：

```text
outputs/checkpoints/sft_lora_small_v2
```

原因是它在 GSM8K-COT limit=20 上达到：

```text
flexible-extract = 0.6000
strict-match     = 0.2000
```

这是当前 small 阶段最好的 flexible-extract 结果。

另一个值得保留的 reward-based format optimization 中间结果是：

```text
outputs/checkpoints/grpo_lora_small_v2_format_reward_384
```

它的评估结果同样是：

```text
flexible-extract = 0.6000
strict-match     = 0.2000
```

但 exact sample comparison 显示它和 `sft_lora_small_v2` 在当前 20 条样本上逐题完全一致：

```text
both_correct = 12
both_wrong = 8
grpo_improved = 0
grpo_regressed = 0
```

因此，当前实际最佳模型仍然优先认为是：

```text
outputs/checkpoints/sft_lora_small_v2
```

`grpo_lora_small_v2_format_reward_384` 应作为 reward-based format optimization 链路跑通的正向中间结果保留。

---

## 3. 当前主要实验结果

### 3.1 small 阶段 GSM8K-COT 结果

| Stage | Sample Len | Flexible Exact Match | Strict Exact Match |
|---|---:|---:|---:|
| sft_lora_small | 20 | 0.4500 | 0.2500 |
| dpo_lora_small | 20 | 0.4000 | 0.2000 |
| grpo_lora_small | 20 | 0.4000 | 0.2000 |
| sft_lora_small_v2 | 20 | 0.6000 | 0.2000 |
| sft_lora_small_v2_format | 20 | 0.3500 | 0.1500 |
| grpo_lora_small_format_reward | 20 | 0.4000 | 0.2000 |
| grpo_lora_small_v2_format_reward_384 | 20 | 0.6000 | 0.2000 |

注意：这些结果都来自本地 CPU 小规模实验，`limit=20`，不能作为正式模型性能结论，只能作为工程链路验证和阶段对比依据。

---

## 4. 正向结果

### 4.1 targeted SFT small_v2 是正向结果

`sft_lora_small_v2` 相比 `sft_lora_small`：

```text
sft_lora_small:
flexible = 0.4500
strict   = 0.2500

sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000
```

结论：

```text
基于错误模式补充 targeted 数据是有效的。
```

样本级对比显示 targeted small_v2 修复了 3 道 SFT small 原本答错的题，并且没有造成回退。

---

### 4.2 prompt-level format eval v2 是正向中间结果

prompt-level format eval v1：

```text
flexible_acc = 0.4000
strict_hash_acc = 0.4000
```

prompt-level format eval v2：

```text
flexible_acc = 0.5500
format_acc = 0.4000
final_answer_format_hit_rate = 0.7000
```

结论：

```text
温和的 Final answer: <answer> 格式提示优于强制 #### <answer>。
```

但它仍然低于原始 `sft_lora_small_v2` 的 flexible=0.6000，因此不是最终方案。

---

### 4.3 final-answer reward 设计是正向工程结果

本项目实现了：

```text
src/rewards/final_answer_reward.py
```

reward 结构为：

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

这个设计可以避免模型只学格式、不学正确答案。

---

### 4.4 reward-based format optimization 链路已经跑通

本阶段已经证明：

```text
1. final-answer reward 可以正常计算；
2. final-answer reward 可以接入 GRPOTrainer；
3. GRPOTrainer 可以训练并保存 adapter；
4. lm-eval 可以加载训练后的 adapter；
5. 可以进行 exact sample comparison。
```

---

### 4.5 从 sft_lora_small_v2 出发优于从 dpo_lora_small 出发

从 `dpo_lora_small` 出发：

```text
grpo_lora_small_format_reward:
flexible = 0.4000
strict   = 0.2000
```

从 `sft_lora_small_v2` 出发，并使用 `max_completion_length=384`：

```text
grpo_lora_small_v2_format_reward_384:
flexible = 0.6000
strict   = 0.2000
```

结论：

```text
起点 checkpoint 很重要。
当前最佳 SFT checkpoint 是更合理的 GRPO 起点。
```

---

### 4.6 max_completion_length=384 是必要改进

从 `sft_lora_small_v2` 出发时：

```text
max_completion_length = 256:
reward_std = 0
frac_reward_zero_std = 1
grad_norm = 0
```

提升到 384 后：

```text
max_completion_length = 384:
reward_std = 0.5774
frac_reward_zero_std = 0
grad_norm = 0.7084
```

结论：

```text
256 对当前 format-reward GRPO 不够，容易截断；
384 能恢复 reward 方差。
```

---

## 5. 负结果和限制

### 5.1 format-constrained SFT small_v2 是负结果

`outputs/checkpoints/sft_lora_small_v2_format` 的结果为：

```text
flexible = 0.3500
strict   = 0.1500
```

相比 `sft_lora_small_v2` 明显下降：

```text
sft_lora_small_v2:
flexible = 0.6000
strict   = 0.2000
```

结论：

```text
直接强制改写 SFT 训练文本格式会破坏模型推理表现。
```

---

### 5.2 prompt-level format eval 不能作为最终方案

prompt-level format eval v2 虽然比 v1 好，但仍然没有恢复到原始 `sft_lora_small_v2` 的 flexible score：

```text
sft_lora_small_v2 flexible = 0.6000
prompt format v2 flexible = 0.5500
```

结论：

```text
只改 prompt 可以提升格式遵循，但仍可能干扰推理。
```

---

### 5.3 当前 GRPO-384 只是追平，没有超过

`grpo_lora_small_v2_format_reward_384` 的结果：

```text
flexible = 0.6000
strict   = 0.2000
```

和 `sft_lora_small_v2` 完全一致。

样本级对比：

```text
both_correct = 12
both_wrong = 8
grpo_improved = 0
grpo_regressed = 0
```

结论：

```text
GRPO-384 没有新增正确样本；
当前不应该继续盲目增加 GRPO step。
```

---

### 5.4 当前结果都不是正式性能结论

原因：

```text
本地环境：CPU
dtype: float32
batch_size: 1
limit: 20
max_steps: 1 / 5 / 10 / 20 / 30
```

这些实验主要用于：

```text
工程链路验证
误差分析
debug / small-scale 方向筛选
```

不能作为正式模型性能。

---

## 6. reward-based format optimization 阶段完整结论

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

---

## 7. 当前最重要的项目结论

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

---

## 8. 当前关键文件

### 8.1 最重要 checkpoint

```text
outputs/checkpoints/sft_lora_small_v2
outputs/checkpoints/grpo_lora_small_v2_format_reward_384
```

### 8.2 reward-based format optimization 相关脚本

```text
src/rewards/final_answer_reward.py
scripts/30_test_final_answer_reward.py
scripts/31_train_grpo_format_reward_debug.py
scripts/32_inspect_grpo_format_reward.py
scripts/33_prepare_grpo_format_reward_data.py
scripts/36_train_grpo_format_reward_from_adapter_debug.py
scripts/37_inspect_sft_v2_format_reward.py
scripts/38_compare_sft_v2_vs_grpo_384_samples.py
```

### 8.3 关键配置

```text
configs/grpo_format_reward_debug.yaml
configs/grpo_format_reward_small.yaml
configs/grpo_format_reward_from_sft_v2_debug.yaml
configs/grpo_format_reward_from_sft_v2_debug_384.yaml
configs/grpo_format_reward_from_sft_v2_384_small.yaml
configs/eval_grpo_sft_v2_format_reward_384_lora.yaml
```

### 8.4 关键报告

```text
outputs/reports/reward_based_format_optimization_stage_summary.md
outputs/reports/grpo_sft_v2_format_reward_384_eval_report.md
outputs/reports/sft_v2_vs_grpo_384_sample_comparison.md
outputs/reports/sft_v2_format_reward_inspection.md
outputs/reports/grpo_format_reward_from_sft_v2_384_small_report.md
README.md
```

---

## 9. 下一阶段最合理方向

当前不建议继续在本地盲目加 GRPO step。

下一阶段有三个合理方向。

---

### 9.1 方向一：整理最终项目报告

这是最稳的方向。

原因：

```text
当前项目已经有完整闭环；
已经有正向结果、负结果和误差分析；
足够形成一份完整项目报告；
适合用于简历、面试和后续开题式讲解。
```

建议下一步做：

```text
outputs/reports/final_project_report.md
```

---

### 9.2 方向二：准备云服务器迁移

如果要继续提升指标，应该迁移到 NVIDIA 云服务器。

云服务器阶段可以扩大：

```text
batch size
max_steps
训练数据量
评估 limit
模型规模
```

建议迁移前先准备：

```text
requirements.txt
运行命令清单
checkpoint / data / outputs 迁移清单
GPU 配置建议
```

---

### 9.3 方向三：加入代码推理分支

简历项目最初目标包含数学与代码推理。

如果要扩展代码推理，下一步应加入：

```text
HumanEval
MBPP
EvalPlus
```

并建立：

```text
代码推理 baseline eval
代码推理 SFT 数据构造
代码推理错误分析
```

---

## 10. 当前建议

当前最推荐的下一步是：

```text
先整理最终项目报告，再决定是否租云服务器。
```

原因：

```text
1. 当前本地实验已经形成完整闭环；
2. 继续盲目训练收益不明确；
3. 最终报告可以帮助判断项目还缺什么；
4. 报告也可以直接转化为简历和面试讲解材料。
```

---

## 11. 当前可写入简历的表述

```text
基于 Qwen2.5-1.5B-Instruct 构建评估驱动的数学推理对齐实验框架，完成 baseline、LoRA SFT、DPO、GRPO/RLVR 多阶段后训练闭环；接入 lm-evaluation-harness，支持 GSM8K-COT 评估、LoRA adapter 评估、样本输出分析、错误类型统计、reasoning 错误模式归因和结果汇总；在本地 CPU 环境下完成 debug、small、targeted small_v2、format-constrained small_v2、prompt-level format eval 与 reward-based format optimization 多级实验验证；通过 targeted 数据将 GSM8K-COT small 评估的 flexible-extract 从 0.4500 提升到 0.6000；进一步设计 final-answer correctness/format/extractability 组合 reward 并接入 TRL GRPOTrainer，验证从 SFT/DPO checkpoint 出发的 reward-based format optimization 全流程；发现从最佳 SFT checkpoint 出发并提升 max_completion_length 到 384 可恢复 reward variance，但当前 5-step GRPO 在 GSM8K-COT limit=20 上追平而未超过 SFT baseline，为后续扩展到更大规模 RLVR、MATH、HumanEval、MBPP 和云端训练打下工程基础。
```