# Evaluation-Driven SFT-DPO-GRPO Reasoning Alignment

本项目基于 Qwen2.5-1.5B/3B-Instruct 构建数学与代码推理对齐系统，参考 Open-R1 推理模型 pipeline，完成评估先行、数据构建、LoRA SFT、DPO、GRPO/RLVR 多阶段后训练实验。

## 项目目标

本项目不是只做微调，而是构建一个评估驱动的后训练闭环：

1. 使用 lm-eval 对原始模型进行 baseline 评估
2. 使用 OpenR1-Math、GSM8K、MATH 构建统一 CoT 数学推理数据
3. 使用 LoRA / PEFT 进行 SFT
4. 使用 distilabel-math-preference-dpo 进行 DPO 偏好对齐
5. 使用 GRPO/RLVR 结合 rule-based reward 优化数学与代码推理
6. 每个阶段后重新评估原始模型或 LoRA adapter
7. 保存 lm-eval samples 和自定义 bad cases，进行错误分析

## 模型

- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct

## 数据集

### 数学训练数据

- open-r1/OpenR1-Math-220k
- openai/gsm8k
- EleutherAI/hendrycks_math

### DPO 数据

- argilla/distilabel-math-preference-dpo

### 后续代码评估

- HumanEval
- MBPP
- EvalPlus

## 项目模块

- lm-eval 标准评估
- 自定义 bad case 分析
- SFT 数据构造
- LoRA SFT
- DPO 偏好对齐
- GRPO/RLVR rule-based reward
- LoRA adapter 评估
- 最终实验报告

## 当前进度

- [x] VS Code 项目结构搭建
- [x] Python 虚拟环境
- [x] lm-eval 接入
- [x] baseline 评估脚本
- [x] LoRA adapter 评估接口
- [x] 自定义 bad case 保存
- [x] 数据集读取检查
- [ ] SFT 数据构造
- [ ] LoRA SFT
- [ ] SFT 后评估
- [ ] DPO 数据构造
- [ ] DPO 训练
- [ ] GRPO/RLVR 训练
- [ ] 最终实验报告