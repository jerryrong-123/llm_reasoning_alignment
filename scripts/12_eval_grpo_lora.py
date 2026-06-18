import os
import subprocess
import sys


CONFIG_PATH = "configs/eval_grpo_lora.yaml"
ADAPTER_CONFIG_PATH = "outputs/checkpoints/grpo_lora/adapter_config.json"


def main():
    if not os.path.exists(ADAPTER_CONFIG_PATH):
        print("没有找到 GRPO LoRA adapter。")
        print(f"缺少文件: {ADAPTER_CONFIG_PATH}")
        print("请先运行:")
        print("python scripts/11_train_grpo.py --config configs/grpo_debug.yaml")
        return

    cmd = [
        sys.executable,
        "scripts/01_lmeval_eval.py",
        "--config",
        CONFIG_PATH,
    ]

    print("====== 即将评估 GRPO LoRA adapter ======")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print("\n====== GRPO LoRA 评估完成 ======")


if __name__ == "__main__":
    main()