import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache/transformers")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


PROJECT_ROOT = Path("/root/autodl-tmp/llm_reasoning_alignment_server_restored")

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag" / "rag_sft_train_2000.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "qwen25_7b_rag_sft_lora_debug"
REPORT_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports" / "qwen25_7b_rag_sft_lora_debug_report.md"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 1024
MAX_STEPS = 20
TRAIN_LIMIT = 128


def load_tokenizer():
    print("====== 加载 tokenizer ======", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_model():
    print("====== 加载 4bit QLoRA model ======", flush=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    print("====== LoRA 参数统计 ======", flush=True)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
        print("memory allocated GB:", torch.cuda.memory_allocated(0) / 1024**3, flush=True)
        print("memory reserved GB:", torch.cuda.memory_reserved(0) / 1024**3, flush=True)

    return model


def build_texts(example, tokenizer):
    messages = example["messages"]

    prompt_messages = messages[:-1]
    full_messages = messages

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return prompt_text, full_text


def tokenize_example(example, tokenizer):
    prompt_text, full_text = build_texts(example, tokenizer)

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt["input_ids"]), len(labels))

    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for x in features:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            labels = x["labels"]

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main():
    print("====== Qwen2.5-7B RAG-SFT LoRA Debug Training ======", flush=True)
    print("model:", MODEL_PATH, flush=True)
    print("train_path:", TRAIN_PATH, flush=True)
    print("output_dir:", OUTPUT_DIR, flush=True)
    print("max_length:", MAX_LENGTH, flush=True)
    print("max_steps:", MAX_STEPS, flush=True)
    print("train_limit:", TRAIN_LIMIT, flush=True)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"训练数据不存在: {TRAIN_PATH}")

    tokenizer = load_tokenizer()
    model = load_model()

    print("====== 加载训练数据 ======", flush=True)

    ds = load_dataset(
        "json",
        data_files=str(TRAIN_PATH),
        split="train",
    )

    ds = ds.select(range(min(TRAIN_LIMIT, len(ds))))

    print("debug train examples:", len(ds), flush=True)

    tokenized = ds.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    print("tokenized columns:", tokenized.column_names, flush=True)
    print("first tokenized length:", len(tokenized[0]["input_ids"]), flush=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_steps=MAX_STEPS,
        save_total_limit=1,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=CausalLMCollator(tokenizer),
    )

    print("====== 开始 debug 训练 ======", flush=True)

    train_result = trainer.train()

    print("====== 保存 LoRA adapter ======", flush=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics

    report_lines = []
    report_lines.append("# Qwen2.5-7B RAG-SFT LoRA Debug Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append("This debug run verifies that Qwen2.5-7B can be fine-tuned with 4bit QLoRA on the RAG-SFT data.")
    report_lines.append("")
    report_lines.append("## 2. Settings")
    report_lines.append("")
    report_lines.append(f"- Model: `{MODEL_PATH}`")
    report_lines.append(f"- Train data: `{TRAIN_PATH}`")
    report_lines.append(f"- Output dir: `{OUTPUT_DIR}`")
    report_lines.append(f"- Train limit: {TRAIN_LIMIT}")
    report_lines.append(f"- Max length: {MAX_LENGTH}")
    report_lines.append(f"- Max steps: {MAX_STEPS}")
    report_lines.append(f"- LoRA r: 16")
    report_lines.append(f"- LoRA alpha: 32")
    report_lines.append(f"- Quantization: 4bit NF4")
    report_lines.append("")
    report_lines.append("## 3. Train Metrics")
    report_lines.append("")
    for k, v in sorted(metrics.items()):
        report_lines.append(f"- {k}: {v}")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== Debug 训练完成 ======", flush=True)
    print("adapter:", OUTPUT_DIR, flush=True)
    print("report:", REPORT_PATH, flush=True)
    print("metrics:", json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    if torch.cuda.is_available():
        print("final memory allocated GB:", torch.cuda.memory_allocated(0) / 1024**3, flush=True)
        print("final memory reserved GB:", torch.cuda.memory_reserved(0) / 1024**3, flush=True)


if __name__ == "__main__":
    main()
