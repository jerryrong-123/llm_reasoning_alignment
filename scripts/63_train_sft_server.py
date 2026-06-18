import argparse
import inspect
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sft_config(cfg):
    """
    兼容不同 TRL 版本的 SFTConfig 参数。
    """
    params = inspect.signature(SFTConfig.__init__).parameters

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

    if "max_length" in params:
        args["max_length"] = int(cfg["max_seq_length"])

    if "max_seq_length" in params:
        args["max_seq_length"] = int(cfg["max_seq_length"])

    if "dataset_text_field" in params:
        args["dataset_text_field"] = "text"

    if "packing" in params:
        args["packing"] = False

    if "fp16" in params:
        args["fp16"] = str(cfg.get("dtype", "bf16")).lower() in ("fp16", "float16")

    if "bf16" in params:
        args["bf16"] = str(cfg.get("dtype", "bf16")).lower() in ("bf16", "bfloat16")

    return SFTConfig(**args)


def build_trainer(model, tokenizer, train_dataset, training_args, lora_config, cfg):
    """
    兼容不同 TRL 版本：
    新版本常用 processing_class；
    老版本可能用 tokenizer / dataset_text_field / max_seq_length。
    """
    sig = inspect.signature(SFTTrainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
    }

    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in sig:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in sig:
        trainer_kwargs["max_seq_length"] = int(cfg["max_seq_length"])

    if "packing" in sig:
        trainer_kwargs["packing"] = False

    return SFTTrainer(**trainer_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_debug.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model = cfg["base_model"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    print("====== 加载 SFT 数据 ======")
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

    print("\n====== 创建 LoRA 配置 ======")
    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("\n====== 创建 SFTConfig ======")
    training_args = build_sft_config(cfg)

    print("\n====== 创建 SFTTrainer ======")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        lora_config=lora_config,
        cfg=cfg,
    )

    print("\n====== 开始 LoRA SFT 训练 ======")
    print("注意：当前 max_steps 很小，只是为了验证训练链路。")
    trainer.train()

    print("\n====== 保存 LoRA adapter ======")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n====== SFT 训练完成 ======")
    print(f"LoRA adapter 保存到: {output_dir}")
    print("后面 lm-eval 会通过 peft_path 加载这个 adapter。")


if __name__ == "__main__":
    main()

