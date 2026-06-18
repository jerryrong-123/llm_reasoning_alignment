import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

try:
    from trl import DPOConfig
except Exception:
    DPOConfig = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}


def get_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if dtype_name in ["fp16", "float16"]:
        return torch.float16
    if dtype_name in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    base_model = cfg["base_model"]
    sft_adapter_path = PROJECT_ROOT / cfg["sft_adapter_path"]
    train_file = PROJECT_ROOT / cfg["train_file"]
    output_dir = PROJECT_ROOT / cfg["output_dir"]

    dtype = get_dtype(cfg.get("dtype", "bf16"))

    print("====== DPO_v4 targeted training ======")
    print("config:", config_path)
    print("base_model:", base_model)
    print("sft_adapter_path:", sft_adapter_path)
    print("train_file:", train_file)
    print("output_dir:", output_dir)

    assert sft_adapter_path.exists(), f"sft_adapter_path not found: {sft_adapter_path}"
    assert train_file.exists(), f"train_file not found: {train_file}"

    print("\n====== 加载 DPO 数据 ======")
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    print(dataset)
    print("字段:", dataset.column_names)

    print("\n====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n====== 加载 policy model: base + SFT_v5 adapter，可训练 ======")
    base_policy = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_policy.config.use_cache = False

    model = PeftModel.from_pretrained(
        base_policy,
        str(sft_adapter_path),
        is_trainable=True,
    )

    print("\n====== 加载 reference model: base + SFT_v5 adapter，冻结 ======")
    base_ref = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_ref.config.use_cache = False

    ref_model = PeftModel.from_pretrained(
        base_ref,
        str(sft_adapter_path),
        is_trainable=False,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    common_args = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        max_steps=int(cfg.get("max_steps", 100)),
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_steps=int(cfg.get("save_steps", 50)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=str(cfg.get("dtype", "bf16")).lower() in ["bf16", "bfloat16"],
        fp16=str(cfg.get("dtype", "bf16")).lower() in ["fp16", "float16"],
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "linear")),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )

    dpo_extra_args = dict(
        beta=float(cfg.get("beta", 0.1)),
        max_length=int(cfg.get("max_length", 768)),
        max_prompt_length=int(cfg.get("max_prompt_length", 320)),
        max_completion_length=int(cfg.get("max_completion_length", 448)),
        loss_type=str(cfg.get("loss_type", "sigmoid")),
    )

    print("\n====== 创建训练参数 ======")
    if DPOConfig is not None:
        config_kwargs = {}
        config_kwargs.update(common_args)
        config_kwargs.update(dpo_extra_args)
        train_args = DPOConfig(**filter_kwargs(DPOConfig, config_kwargs))
    else:
        train_args = TrainingArguments(**filter_kwargs(TrainingArguments, common_args))

    print("max_steps:", common_args["max_steps"])
    print("learning_rate:", common_args["learning_rate"])
    print("beta:", dpo_extra_args["beta"])
    print("max_length:", dpo_extra_args["max_length"])
    print("max_prompt_length:", dpo_extra_args["max_prompt_length"])
    print("max_completion_length:", dpo_extra_args["max_completion_length"])

    print("\n====== 创建 DPOTrainer ======")
    trainer_sig = inspect.signature(DPOTrainer.__init__)
    trainer_params = set(trainer_sig.parameters.keys())

    trainer_kwargs = dict(
        model=model,
        ref_model=ref_model,
        args=train_args,
        train_dataset=dataset,
    )

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # 老版本 TRL 可能需要这些参数直接传给 DPOTrainer
    for k, v in dpo_extra_args.items():
        if k in trainer_params:
            trainer_kwargs[k] = v

    trainer = DPOTrainer(**trainer_kwargs)

    print("\n====== 开始 DPO_v4 targeted 训练 ======")
    trainer.train()

    print("\n====== 保存 DPO_v4 adapter ======")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\n====== 完成 ======")
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
