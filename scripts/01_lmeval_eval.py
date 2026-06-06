import argparse
import os
import subprocess
import sys
import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model = cfg["base_model"]
    peft_path = cfg.get("peft_path", None)
    tasks = cfg["tasks"]
    limit = str(cfg.get("limit", 5))
    batch_size = str(cfg.get("batch_size", 1))
    device = cfg.get("device", "cpu")
    dtype = cfg.get("dtype", "float32")
    output_name = cfg["output_name"]
    apply_chat_template = bool(cfg.get("apply_chat_template", True))

    os.makedirs("outputs/eval", exist_ok=True)

    output_path = os.path.join("outputs", "eval", output_name)

    model_args = f"pretrained={base_model},dtype={dtype}"

    if peft_path not in [None, "null", ""]:
        model_args += f",peft={peft_path}"

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--device",
        device,
        "--batch_size",
        batch_size,
        "--limit",
        limit,
        "--output_path",
        output_path,
        "--log_samples",
    ]

    if apply_chat_template:
        cmd.append("--apply_chat_template")

    print("====== 即将运行 lm-eval ======")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print("\n====== lm-eval 运行完成 ======")
    print("结果目录:", output_path)


if __name__ == "__main__":
    main()