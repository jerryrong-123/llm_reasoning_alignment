import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
OUTPUT_PATH = OUTPUT_DIR / "public_game_feedback_prompt_llm_v2_predictions.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "public_game_feedback_prompt_llm_v2_summary.json"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
ALLOWED_TOPICS = {
    "bugs",
    "company_reputation",
    "gameplay",
    "graphics",
    "monetization",
    "multiplayer",
    "other",
    "performance",
    "price",
    "story",
    "updates_support",
}


TOPIC_DEFINITIONS = """
Topic definitions:
- bugs: crashes, glitches, broken features, errors, things not working.
- performance: FPS, lag, stutter, loading speed, latency, optimization, server performance.
- gameplay: combat, mechanics, controls, balance, difficulty, missions, quests, level design.
- graphics: visuals, art style, textures, animation, lighting, visual quality.
- story: plot, lore, characters, dialogue, writing, ending, narrative.
- monetization: microtransactions, skins, gacha, loot boxes, battle pass, pay-to-win.
- price: price, discount, sale, refund, value for money, whether something is worth buying.
- multiplayer: online play, co-op, PvP, matchmaking, ranked games, team play, lobbies.
- updates_support: patches, updates, developer support, bug fixes, roadmap.
- company_reputation: publisher/studio/company reputation, trust toward a game company.
- other: use this when the comment is about a general recommendation, unclear topic, or none of the above.
"""


SENTIMENT_DEFINITIONS = """
Sentiment definitions:
- positive: clear praise, enjoyment, recommendation, or favorable opinion.
- negative: clear complaint, disappointment, criticism, or unfavorable opinion.
- neutral: factual question, recommendation request, objective statement, or no clear emotion.
- mixed: both clear positive and clear negative opinions are present.
"""


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_prompt(text: str) -> str:
    return f"""You are a careful game community feedback analyst.

Your task is to classify one gaming community comment.

{SENTIMENT_DEFINITIONS}

{TOPIC_DEFINITIONS}

Important rules:
1. Do not over-infer emotion. If the comment is mainly a question or factual statement, choose neutral.
2. If multiple topics appear, choose the most central topic of the comment.
3. If the topic is unclear or only a generic recommendation, choose other.
4. Return only valid JSON. Do not add markdown. Do not add extra text.

Return this JSON schema exactly:
{{
  "sentiment": "positive | negative | neutral | mixed",
  "topic": "bugs | company_reputation | gameplay | graphics | monetization | multiplayer | other | performance | price | story | updates_support",
  "reasoning": "short reason",
  "confidence": 0.0
}}

Comment:
{text}
"""


def extract_json(text: str):
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return None

    candidate = match.group(0)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_prediction(obj):
    if not isinstance(obj, dict):
        return {
            "sentiment": "neutral",
            "topic": "other",
            "reasoning": "Failed to parse model output.",
            "confidence": 0.0,
            "parse_ok": False,
        }

    sentiment = str(obj.get("sentiment", "neutral")).strip().lower()
    topic = str(obj.get("topic", "other")).strip().lower()
    reasoning = str(obj.get("reasoning", "")).strip()

    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if sentiment not in ALLOWED_SENTIMENTS:
        sentiment = "neutral"

    if topic not in ALLOWED_TOPICS:
        topic = "other"

    confidence = max(0.0, min(confidence, 1.0))

    return {
        "sentiment": sentiment,
        "topic": topic,
        "reasoning": reasoning,
        "confidence": confidence,
        "parse_ok": True,
    }


def accuracy(rows, expected_key: str, pred_key: str):
    if not rows:
        return 0.0

    correct = sum(1 for row in rows if row.get(expected_key) == row.get(pred_key))
    return correct / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    print("====== Prompt-based LLM game feedback classification v2 ======")
    print(f"model: {MODEL_NAME}")
    print(f"input_path: {INPUT_PATH}")
    print(f"output_path: {OUTPUT_PATH}")
    print(f"summary_path: {SUMMARY_PATH}")
    print(f"limit: {args.limit}")
    print("注意：这是 Prompt v2 baseline，不是规则版 baseline。")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.to(args.device)
    model.eval()

    predictions = []

    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row["text"])

        messages = [
            {
                "role": "system",
                "content": "You are a careful assistant that returns only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(args.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = extract_json(raw_output)
        normalized = normalize_prediction(parsed)

        item = dict(row)
        item["raw_model_output"] = raw_output
        item["pred_sentiment"] = normalized["sentiment"]
        item["pred_topic"] = normalized["topic"]
        item["pred_reasoning"] = normalized["reasoning"]
        item["pred_confidence"] = normalized["confidence"]
        item["parse_ok"] = normalized["parse_ok"]
        item["sentiment_correct"] = item["pred_sentiment"] == row.get("expected_sentiment")
        item["topic_correct"] = item["pred_topic"] == row.get("expected_topic")

        predictions.append(item)

        print(
            f"[{idx}/{len(rows)}] "
            f"id={row.get('id')} "
            f"sentiment={item['pred_sentiment']} "
            f"topic={item['pred_topic']} "
            f"parse_ok={item['parse_ok']}"
        )

    sentiment_acc = accuracy(predictions, "expected_sentiment", "pred_sentiment")
    topic_acc = accuracy(predictions, "expected_topic", "pred_topic")
    parse_rate = sum(1 for row in predictions if row["parse_ok"]) / len(predictions)

    summary = {
        "model": MODEL_NAME,
        "limit": args.limit,
        "sample_count": len(predictions),
        "sentiment_acc": round(sentiment_acc, 4),
        "topic_acc": round(topic_acc, 4),
        "parse_rate": round(parse_rate, 4),
        "baseline_type": "prompt_based_llm_v2",
        "notes": [
            "Prompt v2 adds explicit sentiment and topic definitions.",
            "The public dataset labels may contain noise because the dataset uses LLM-assisted annotation.",
            "This result should be compared with rule baseline and prompt v1 baseline on the same samples.",
        ],
    }

    write_jsonl(OUTPUT_PATH, predictions)
    write_json(SUMMARY_PATH, summary)

    print("====== 运行完成 ======")
    print(f"sample_count: {len(predictions)}")
    print(f"sentiment_acc: {sentiment_acc:.4f}")
    print(f"topic_acc: {topic_acc:.4f}")
    print(f"parse_rate: {parse_rate:.4f}")


if __name__ == "__main__":
    main()