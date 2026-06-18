from datasets import load_dataset


def preview_dataset(name, split="train[:2]", config=None):
    print("\n" + "=" * 100)
    print(f"数据集: {name}")
    print("=" * 100)

    if config is None:
        ds = load_dataset(name, split=split)
    else:
        ds = load_dataset(name, config, split=split)

    print(ds)
    print("字段:", ds.column_names)

    for i in range(len(ds)):
        print(f"\n--- 样本 {i} ---")
        print(ds[i])

    return ds


def main():
    preview_dataset(
        name="openai/gsm8k",
        config="main",
        split="train[:2]",
    )

    preview_dataset(
        name="open-r1/OpenR1-Math-220k",
        split="train[:2]",
    )

    preview_dataset(
        name="argilla/distilabel-math-preference-dpo",
        split="train[:2]",
    )

    # MATH 数据集按子领域加载，先看 algebra
    preview_dataset(
        name="EleutherAI/hendrycks_math",
        config="algebra",
        split="test[:2]",
    )

    print("\n====== 数据集读取检查完成 ======")


if __name__ == "__main__":
    main()