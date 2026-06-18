import argparse
import inspect
import os

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dpo_config(cfg):
    """
    兼容不同 TRL 版本的 DPOConfig 参数。
    """
    params = inspect.signature(DPOConfig.__init__).parameters

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

    if "beta" in params:
        args["beta"] = float(cfg["beta"])

    if "max_length" in params:
        args["max_length"] = int(cfg["max_length"])

    if "max_prompt_length" in params:
        args["max_prompt_length"] = int(cfg["max_prompt_length"])

    if "max_completion_length" in params:
        args["max_completion_length"] = int(cfg["max_completion_length"])

    if "fp16" in params:
        args["fp16"] = False

    if "bf16" in params:
        args["bf16"] = False

    return DPOConfig(**args)


def build_dpo_trainer(model, tokenizer, train_dataset, training_args):
    """
    兼容不同 TRL 版本：
    新版本通常用 processing_class；
    老版本可能用 tokenizer。
    """
    sig = inspect.signature(DPOTrainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if "ref_model" in sig:
        trainer_kwargs["ref_model"] = None

    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    return DPOTrainer(**trainer_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dpo_debug.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model = cfg["base_model"]
    sft_adapter_path = cfg["sft_adapter_path"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]

    if not os.path.exists(sft_adapter_path):
        raise FileNotFoundError(
            f"没有找到 SFT adapter: {sft_adapter_path}\n"
            f"请先运行: python scripts/05_train_sft.py --config configs/sft_debug.yaml"
        )

    os.makedirs(output_dir, exist_ok=True)

    print("====== 加载 DPO 数据 ======")
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

    print("\n====== 加载 SFT LoRA adapter，并设为可训练 ======")
    model = PeftModel.from_pretrained(
        model,
        sft_adapter_path,
        is_trainable=True,
    )

    print("\n====== 创建 DPOConfig ======")
    training_args = build_dpo_config(cfg)

    print("\n====== 创建 DPOTrainer ======")
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
    )

    print("\n====== 开始 DPO 训练 ======")
    print("注意：当前 max_steps=1，只是为了验证 DPO 链路。")
    trainer.train()

    print("\n====== 保存 DPO LoRA adapter ======")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n====== DPO 训练完成 ======")
    print(f"DPO LoRA adapter 保存到: {output_dir}")
    print("后面 lm-eval 会通过 peft_path 加载这个 adapter。")


if __name__ == "__main__":
    main()