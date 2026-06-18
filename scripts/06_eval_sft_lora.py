import os
import subprocess
import sys


CONFIG_PATH = "configs/eval_sft_lora.yaml"
ADAPTER_CONFIG_PATH = "outputs/checkpoints/sft_lora/adapter_config.json"


def main():
    if not os.path.exists(ADAPTER_CONFIG_PATH):
        print("没有找到 SFT LoRA adapter。")
        print(f"缺少文件: {ADAPTER_CONFIG_PATH}")
        print("请先运行:")
        print("python scripts/05_train_sft.py --config configs/sft_debug.yaml")
        return

    cmd = [
        sys.executable,
        "scripts/01_lmeval_eval.py",
        "--config",
        CONFIG_PATH,
    ]

    print("====== 即将评估 SFT LoRA adapter ======")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print("\n====== SFT LoRA 评估完成 ======")


if __name__ == "__main__":
    main()