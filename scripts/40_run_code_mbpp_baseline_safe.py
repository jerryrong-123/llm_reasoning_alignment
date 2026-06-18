import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_simple_yaml(config_path: Path) -> dict:
    """
    只解析当前项目这种简单 key: value YAML。
    这样可以避免额外依赖 PyYAML，减少环境问题。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = {}

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value.lower() in {"null", "none"}:
            parsed_value = None
        elif value.lower() == "true":
            parsed_value = True
        elif value.lower() == "false":
            parsed_value = False
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value.strip('"').strip("'")

        config[key] = parsed_value

    return config


def build_lm_eval_command(config: dict) -> tuple[list[str], Path, Path]:
    base_model = config["base_model"]
    peft_path = config.get("peft_path")
    tasks = config["tasks"]
    limit = str(config.get("limit", 5))
    batch_size = str(config.get("batch_size", 1))
    device = config.get("device", "cpu")
    dtype = config.get("dtype", "float32")
    apply_chat_template = bool(config.get("apply_chat_template", True))
    predict_only = bool(config.get("predict_only", False))
    output_name = config["output_name"]

    output_dir = PROJECT_ROOT / "outputs" / "eval" / output_name
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_file = log_dir / f"{output_name}.log"

    model_args_parts = [
        f"pretrained={base_model}",
        f"dtype={dtype}",
        "trust_remote_code=True",
    ]

    if peft_path:
        model_args_parts.append(f"peft={peft_path}")

    model_args = ",".join(model_args_parts)

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--limit",
        limit,
        "--batch_size",
        batch_size,
        "--device",
        device,
        "--num_fewshot",
        "0",
        "--output_path",
        str(output_dir),
        "--log_samples",
    ]

    if apply_chat_template:
        cmd.append("--apply_chat_template")

    if predict_only:
        cmd.append("--predict_only")

    return cmd, output_dir, log_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行 lm-eval。默认不执行，只打印将要运行的命令。",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "eval_code_baseline_mbpp.yaml"
    config = parse_simple_yaml(config_path)

    cmd, output_dir, log_file = build_lm_eval_command(config)

    print("====== MBPP baseline eval 安全运行脚本 ======")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"配置文件: {config_path}")
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file}")
    print()
    print("将要运行的命令:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    if config.get("predict_only", False):
        print("当前配置是 predict_only=true：")
        print("- 会保存模型生成 sample")
        print("- 不计算 pass@1")
        print("- 不执行模型生成的 Python 代码")
        print()

    if not args.execute:
        print("当前是 dry-run：没有真正运行 lm-eval。")
        print("下一步确认无误后，再加 --execute 真正运行。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print("开始执行 lm-eval。终端不会刷大量输出，详细日志会写入 log 文件。")
    print(f"日志文件: {log_file}")

    with log_file.open("w", encoding="utf-8") as f:
        f.write("====== MBPP baseline eval command ======\n")
        f.write(" ".join(cmd))
        f.write("\n\n====== lm-eval output ======\n")
        f.flush()

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=60 * 60,
        )

    print(f"lm-eval 运行结束，return code = {result.returncode}")
    print(f"请查看日志: {log_file}")
    print(f"请查看输出目录: {output_dir}")


if __name__ == "__main__":
    main()