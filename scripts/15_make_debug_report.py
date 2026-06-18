from pathlib import Path
from datetime import datetime


REPORT_DIR = Path("outputs/reports")
REPORT_PATH = REPORT_DIR / "debug_report.md"


CHECK_ITEMS = {
    "SFT 数据文件": Path("data/processed/sft_debug.jsonl"),
    "DPO 数据文件": Path("data/processed/dpo_debug.jsonl"),
    "GRPO 数据文件": Path("data/processed/grpo_debug.jsonl"),

    "SFT LoRA adapter": Path("outputs/checkpoints/sft_lora/adapter_config.json"),
    "DPO LoRA adapter": Path("outputs/checkpoints/dpo_lora/adapter_config.json"),
    "GRPO LoRA adapter": Path("outputs/checkpoints/grpo_lora/adapter_config.json"),

    "评估汇总 CSV": Path("outputs/reports/eval_summary.csv"),
    "评估汇总 Markdown": Path("outputs/reports/eval_summary.md"),
    "lm-eval 样本预览": Path("outputs/reports/lmeval_samples_preview.jsonl"),
}


def exists_mark(path: Path) -> str:
    return "✅" if path.exists() else "❌"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Debug Stage Report")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 1. 当前项目定位")
    lines.append("")
    lines.append("本项目是一个评估驱动的大模型数学与代码推理对齐系统。")
    lines.append("当前 debug 阶段已经跑通完整训练与评估主链路：")
    lines.append("")
    lines.append("```text")
    lines.append("Baseline 评估")
    lines.append("→ SFT 数据构造")
    lines.append("→ LoRA SFT")
    lines.append("→ SFT 后评估")
    lines.append("→ DPO 数据构造")
    lines.append("→ DPO 训练")
    lines.append("→ DPO 后评估")
    lines.append("→ GRPO/RLVR 数据构造")
    lines.append("→ GRPO/RLVR 训练")
    lines.append("→ GRPO 后评估")
    lines.append("→ 评估结果汇总")
    lines.append("→ 样本输出检查")
    lines.append("```")
    lines.append("")

    lines.append("## 2. 文件检查")
    lines.append("")
    lines.append("| 项目项 | 状态 | 路径 |")
    lines.append("|---|---|---|")

    for name, path in CHECK_ITEMS.items():
        lines.append(f"| {name} | {exists_mark(path)} | `{path}` |")

    lines.append("")

    lines.append("## 3. 当前阶段结论")
    lines.append("")
    lines.append("当前阶段的目标不是获得真实高分，而是验证完整训练和评估链路是否跑通。")
    lines.append("")
    lines.append("当前设置通常包括：")
    lines.append("")
    lines.append("```text")
    lines.append("SFT max_steps = 1")
    lines.append("DPO max_steps = 1")
    lines.append("GRPO max_steps = 1")
    lines.append("lm-eval limit = 5")
    lines.append("device = cpu")
    lines.append("```")
    lines.append("")
    lines.append("因此当前结果只能说明流程正确，不能作为正式模型性能。")
    lines.append("")

    lines.append("## 4. 已形成的项目证据链")
    lines.append("")
    lines.append("当前项目已经具备以下证据：")
    lines.append("")
    lines.append("- 原始模型 baseline 评估结果")
    lines.append("- SFT 后 LoRA adapter")
    lines.append("- SFT 后 lm-eval 结果")
    lines.append("- DPO 后 LoRA adapter")
    lines.append("- DPO 后 lm-eval 结果")
    lines.append("- GRPO 后 LoRA adapter")
    lines.append("- GRPO 后 lm-eval 结果")
    lines.append("- lm-eval 样本级输出")
    lines.append("- 自定义 bad case 文件")
    lines.append("- 评估结果汇总表")
    lines.append("")

    lines.append("## 5. 下一阶段计划")
    lines.append("")
    lines.append("下一阶段要从 debug 版升级到正式小实验版：")
    lines.append("")
    lines.append("1. 扩大 SFT 数据规模，例如 42 条 → 1k 条。")
    lines.append("2. 扩大训练步数，例如 max_steps=1 → 50 / 100。")
    lines.append("3. 扩大 lm-eval 评估数量，例如 limit=5 → 50 / 100。")
    lines.append("4. 增加 MATH / MATH-500 评估。")
    lines.append("5. 增加代码推理评估 HumanEval / MBPP / EvalPlus。")
    lines.append("6. 根据 bad case 分析改进 prompt、答案抽取、reward 函数。")
    lines.append("")

    eval_md = Path("outputs/reports/eval_summary.md")
    if eval_md.exists():
        lines.append("## 6. 当前评估汇总")
        lines.append("")
        lines.append(eval_md.read_text(encoding="utf-8"))

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("====== Debug 阶段报告生成完成 ======")
    print(f"报告路径: {REPORT_PATH}")


if __name__ == "__main__":
    main()