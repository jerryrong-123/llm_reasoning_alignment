# Debug Stage Report

生成时间：2026-06-06 16:31:48

## 1. 当前项目定位

本项目是一个评估驱动的大模型数学与代码推理对齐系统。
当前 debug 阶段已经跑通完整训练与评估主链路：

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
→ 评估结果汇总
→ 样本输出检查
```

## 2. 文件检查

| 项目项 | 状态 | 路径 |
|---|---|---|
| SFT 数据文件 | ✅ | `data\processed\sft_debug.jsonl` |
| DPO 数据文件 | ✅ | `data\processed\dpo_debug.jsonl` |
| GRPO 数据文件 | ✅ | `data\processed\grpo_debug.jsonl` |
| SFT LoRA adapter | ✅ | `outputs\checkpoints\sft_lora\adapter_config.json` |
| DPO LoRA adapter | ✅ | `outputs\checkpoints\dpo_lora\adapter_config.json` |
| GRPO LoRA adapter | ✅ | `outputs\checkpoints\grpo_lora\adapter_config.json` |
| 评估汇总 CSV | ✅ | `outputs\reports\eval_summary.csv` |
| 评估汇总 Markdown | ✅ | `outputs\reports\eval_summary.md` |
| lm-eval 样本预览 | ✅ | `outputs\reports\lmeval_samples_preview.jsonl` |

## 3. 当前阶段结论

当前阶段的目标不是获得真实高分，而是验证完整训练和评估链路是否跑通。

当前设置通常包括：

```text
SFT max_steps = 1
DPO max_steps = 1
GRPO max_steps = 1
lm-eval limit = 5
device = cpu
```

因此当前结果只能说明流程正确，不能作为正式模型性能。

## 4. 已形成的项目证据链

当前项目已经具备以下证据：

- 原始模型 baseline 评估结果
- SFT 后 LoRA adapter
- SFT 后 lm-eval 结果
- DPO 后 LoRA adapter
- DPO 后 lm-eval 结果
- GRPO 后 LoRA adapter
- GRPO 后 lm-eval 结果
- lm-eval 样本级输出
- 自定义 bad case 文件
- 评估结果汇总表

## 5. 下一阶段计划

下一阶段要从 debug 版升级到正式小实验版：

1. 扩大 SFT 数据规模，例如 42 条 → 1k 条。
2. 扩大训练步数，例如 max_steps=1 → 50 / 100。
3. 扩大 lm-eval 评估数量，例如 limit=5 → 50 / 100。
4. 增加 MATH / MATH-500 评估。
5. 增加代码推理评估 HumanEval / MBPP / EvalPlus。
6. 根据 bad case 分析改进 prompt、答案抽取、reward 函数。

## 6. 当前评估汇总

# Evaluation Summary

> 注意：当前结果来自 debug 设置，例如 limit=5、max_steps=1，只能证明流程跑通，不能作为正式模型性能。

| Stage | Task | Metric | Value | Stderr |
|---|---|---|---:|---:|
| baseline | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| baseline | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| baseline | gsm8k_cot | exact_match,strict-match | 0.6000 | 0.2449 |
| baseline | gsm8k_cot | exact_match,strict-match | 0.6000 | 0.2449 |
| baseline | gsm8k_cot | sample_len | 5 | None |
| baseline | gsm8k_cot | sample_len | 5 | None |
| sft_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| sft_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| sft_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| sft_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| sft_lora | gsm8k_cot | sample_len | 5 | None |
| sft_lora | gsm8k_cot | sample_len | 5 | None |
| dpo_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| dpo_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| dpo_lora | gsm8k_cot | sample_len | 5 | None |
| grpo_lora | gsm8k_cot | exact_match,flexible-extract | 0.8000 | 0.2000 |
| grpo_lora | gsm8k_cot | exact_match,strict-match | 0.4000 | 0.2449 |
| grpo_lora | gsm8k_cot | sample_len | 5 | None |