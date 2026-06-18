import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import inspect
import os

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.rewards import compute_final_answer_reward, extract_gold_answer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def completion_to_text(completion) -> str:
    """
    兼容 TRL 可能返回的不同 completion 格式：
    1. 字符串
    2. [{"role": "assistant", "content": "..."}]
    3. 其他对象
    """
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(completion)


def normalize_gold_answer(gold) -> str:
    """
    兼容两种 GRPO 数据：
    1. answer 字段已经是最终答案，例如 42
    2. answer 字段是 GSM8K 原始格式，例如 reasoning #### 42
    """
    if gold is None:
        return ""

    gold_text = str(gold)
    extracted = extract_gold_answer(gold_text)

    if extracted is not None:
        return str(extracted)

    return gold_text


def final_answer_reward_func(completions, answer=None, **kwargs):
    """
    GRPO reward function.

    核心原则：
    answer correctness reward > final-answer format reward

    返回值必须是 list[float]，每个 completion 一个 reward。
    """
    rewards = []

    if answer is None:
        answer = kwargs.get("answers", None)

    if answer is None:
        answer = kwargs.get("gold_answer", None)

    if answer is None:
        answer = [""] * len(completions)

    for completion, gold in zip(completions, answer):
        text = completion_to_text(completion)
        gold_answer = normalize_gold_answer(gold)

        reward_info = compute_final_answer_reward(
            response=text,
            gold_answer=gold_answer,
        )

        rewards.append(float(reward_info["total_reward"]))

    return rewards


def build_grpo_config(cfg):
    """
    兼容不同 TRL 版本的 GRPOConfig 参数。
    """
    params = inspect.signature(GRPOConfig.__init__).parameters

    args = {
        "output_dir": cfg["output_dir"],
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "max_steps": int(cfg["max_steps"]),
        "logging_steps": int(cfg["logging_steps"]),
        "save_steps": int(cfg["save_steps"]),
        "save_total_limit": int(cfg["save_total_limit"]),
        "report_to": [],
        "remove_unused_columns": False,
    }

    if "max_prompt_length" in params:
        args["max_prompt_length"] = int(cfg["max_prompt_length"])

    if "max_completion_length" in params:
        args["max_completion_length"] = int(cfg["max_completion_length"])

    if "num_generations" in params:
        args["num_generations"] = int(cfg["num_generations"])

    if "fp16" in params:
        args["fp16"] = False

    if "bf16" in params:
        args["bf16"] = False

    if "use_vllm" in params:
        args["use_vllm"] = False

    if "beta" in params:
        args["beta"] = 0.0

    if "temperature" in params:
        args["temperature"] = 0.7

    return GRPOConfig(**args)


def build_grpo_trainer(model, tokenizer, train_dataset, training_args):
    """
    兼容不同 TRL 版本的 GRPOTrainer。
    """
    sig = inspect.signature(GRPOTrainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if "reward_funcs" in sig:
        trainer_kwargs["reward_funcs"] = final_answer_reward_func
    elif "reward_func" in sig:
        trainer_kwargs["reward_func"] = final_answer_reward_func
    else:
        raise TypeError("当前 TRL 的 GRPOTrainer 没有 reward_funcs / reward_func 参数，请检查 TRL 版本。")

    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    return GRPOTrainer(**trainer_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_format_reward_debug.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model = cfg["base_model"]
    dpo_adapter_path = cfg["dpo_adapter_path"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]

    if not os.path.exists(dpo_adapter_path):
        raise FileNotFoundError(
            f"没有找到 DPO adapter: {dpo_adapter_path}\n"
            f"请先确认 small DPO adapter 是否存在。"
        )

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"没有找到 GRPO 训练数据: {train_file}\n"
            f"请先确认 data/processed/grpo_small.jsonl 是否存在。"
        )

    os.makedirs(output_dir, exist_ok=True)

    print("====== 加载 GRPO/RLVR format-reward debug 数据 ======")
    train_dataset = load_dataset(
        "json",
        data_files=train_file,
        split="train",
    )

    print(train_dataset)
    print("字段:", train_dataset.column_names)

    print("\n====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n====== 加载 base model ======")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    print("\n====== 加载 DPO LoRA adapter，并设为可训练 ======")
    model = PeftModel.from_pretrained(
        model,
        dpo_adapter_path,
        is_trainable=True,
    )

    print("\n====== 创建 GRPOConfig ======")
    training_args = build_grpo_config(cfg)

    print("\n====== 创建 GRPOTrainer ======")
    trainer = build_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
    )

    print("\n====== 开始 GRPO/RLVR format-reward debug 训练 ======")
    print("注意：当前 max_steps=1，只是为了验证 final-answer reward 接入 GRPO 链路。")
    trainer.train()

    print("\n====== 保存 GRPO format-reward LoRA adapter ======")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n====== GRPO/RLVR format-reward debug 完成 ======")
    print(f"GRPO format-reward LoRA adapter 保存到: {output_dir}")


if __name__ == "__main__":
    main()