import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class CompletionOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in features]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in features]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_completion_only(example, tokenizer, max_seq_length: int):
    prompt = example["prompt"]
    completion = example["completion"]

    if tokenizer.eos_token is None:
        eos = ""
    else:
        eos = tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion + eos, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids

    labels = [-100] * len(prompt_ids) + completion_ids[:]

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_model = cfg["base_model"]
    train_file = PROJECT_ROOT / cfg["train_file"]
    output_dir = PROJECT_ROOT / cfg["output_dir"]

    dtype_name = str(cfg.get("dtype", "bf16")).lower()
    torch_dtype = DTYPE_MAP[dtype_name]

    print("====== SFT v5 completion-only 训练配置 ======")
    print(f"base_model: {base_model}")
    print(f"train_file: {train_file}")
    print(f"output_dir: {output_dir}")
    print(f"dtype: {dtype_name}")
    print(f"max_seq_length: {cfg.get('max_seq_length', 512)}")
    print(f"max_steps: {cfg.get('max_steps', 300)}")
    print(f"learning_rate: {cfg.get('learning_rate', 1e-5)}")

    print("\n====== 加载数据 ======")
    dataset = load_dataset("json", data_files=str(train_file), split="train")
    print(dataset)
    print("字段:", dataset.column_names)

    print("\n====== 加载 tokenizer ======")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n====== 加载 base model ======")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    print("\n====== 注入 LoRA ======")
    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_seq_length = int(cfg.get("max_seq_length", 512))

    print("\n====== tokenizer 数据：prompt mask，completion 算 loss ======")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_completion_only(x, tokenizer, max_seq_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing completion-only SFT data",
    )

    collator = CompletionOnlyCollator(tokenizer)

    fp16 = dtype_name in ("fp16", "float16")
    bf16 = dtype_name in ("bf16", "bfloat16")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=int(cfg.get("max_steps", 300)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_steps=int(cfg.get("save_steps", 50)),
        save_total_limit=int(cfg.get("save_total_limit", 8)),
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    print("\n====== 开始训练 SFT v5 completion-only ======")
    trainer.train()

    print("\n====== 保存 LoRA adapter ======")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\n====== SFT v5 completion-only 训练完成 ======")
    print(f"LoRA adapter 保存到: {cfg['output_dir']}")
    print("后面 lm-eval 会通过 peft_path 加载这个 adapter。")


if __name__ == "__main__":
    main()