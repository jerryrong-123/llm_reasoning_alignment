import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
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
from src.rewards.grpo_math_reward_v5_exact import gsm8k_answer_reward_v5_exact


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_grpo_config(cfg):
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
        args["fp16"] = str(cfg.get("dtype", "bf16")).lower() in ("fp16", "float16")

    if "bf16" in params:
        args["bf16"] = str(cfg.get("dtype", "bf16")).lower() in ("bf16", "bfloat16")

    if "use_vllm" in params:
        args["use_vllm"] = False

    if "beta" in params:
        args["beta"] = float(cfg.get("beta", 0.03))

    if "temperature" in params:
        args["temperature"] = float(cfg.get("temperature", 0.8))

    return GRPOConfig(**args)


def build_grpo_trainer(model, tokenizer, train_dataset, training_args):
    sig = inspect.signature(GRPOTrainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if "reward_funcs" in sig:
        trainer_kwargs["reward_funcs"] = gsm8k_answer_reward_v5_exact
    elif "reward_func" in sig:
        trainer_kwargs["reward_func"] = gsm8k_answer_reward_v5_exact
    else:
        raise TypeError("当前 TRL 的 GRPOTrainer 没有 reward_funcs / reward_func 参数。")

    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    return GRPOTrainer(**trainer_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_v5_from_sft_v5_mixed_replay_300.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model = cfg["base_model"]
    start_adapter_path = cfg["start_adapter_path"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]

    if not os.path.exists(start_adapter_path):
        raise FileNotFoundError(f"没有找到起始 LoRA adapter: {start_adapter_path}")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"没有找到 GRPO 训练数据: {train_file}")

    os.makedirs(output_dir, exist_ok=True)

    print("====== 加载 GRPO_v5 hard-prompt 数据 ======")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    print(train_dataset)
    print("字段:", train_dataset.column_names)

    print("\n====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n====== 加载 base model ======")
    model =AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype={
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }[str(cfg.get("dtype", "bf16")).lower()],
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print("\n====== 加载 SFT_v5 LoRA adapter，并设为可训练 ======")
    print(f"start_adapter_path: {start_adapter_path}")
    model = PeftModel.from_pretrained(
        model,
        start_adapter_path,
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

    print("\n====== 开始 GRPO_v5 strict reward 训练 ======")
    print(f"config: {args.config}")
    print(f"max_steps: {cfg['max_steps']}")
    print(f"learning_rate: {cfg['learning_rate']}")
    print(f"beta: {cfg.get('beta', 0.03)}")
    print(f"num_generations: {cfg['num_generations']}")
    print(f"max_completion_length: {cfg['max_completion_length']}")

    trainer.train()

    print("\n====== 保存 GRPO_v5 LoRA adapter ======")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n====== GRPO_v5 训练完成 ======")
    print(f"adapter 保存到: {output_dir}")


if __name__ == "__main__":
    main()
