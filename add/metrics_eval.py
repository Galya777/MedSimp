import json
import os
from pathlib import Path
from typing import List, Dict

from inference import load_model, generate_response, _passes_safety
from structure import extract_sections
from litteracy import readability_scores


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_validation.jsonl"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "metrics_results.csv"


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def eval_one(tokenizer, model, text: str, profile: str) -> Dict[str, float]:
    sections = extract_sections(text)
    result = generate_response(tokenizer, model, sections, profile)
    before = readability_scores(text)
    after = readability_scores(result)
    length_ratio = (len(result) + 1) / (len(text) + 1)
    original_concat = "\n".join(v for v in sections.values() if v)
    numbers_ok = _passes_safety(original_concat, result)
    return {
        "fkgl_before": before.get("fkgl", -1.0),
        "fkgl_after": after.get("fkgl", -1.0),
        "smog_before": before.get("smog", -1.0),
        "smog_after": after.get("smog", -1.0),
        "length_ratio": float(length_ratio),
        "numbers_preserved": 1.0 if numbers_ok else 0.0,
        "output": result,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading model...")
    tokenizer, model = load_model()

    rows = load_jsonl(DATA_PATH)
    print(f"Loaded {len(rows)} examples from {DATA_PATH}")

    results: List[Dict] = []

    for i, row in enumerate(rows, 1):
        text = row["text"].strip()
        profile = row.get("profile", "low_literacy")
        print(f"\n[{i}/{len(rows)}] profile={profile}")
        metrics = eval_one(tokenizer, model, text, profile)
        results.append({
            "idx": i,
            "profile": profile,
            "fkgl_before": metrics["fkgl_before"],
            "fkgl_after": metrics["fkgl_after"],
            "smog_before": metrics["smog_before"],
            "smog_after": metrics["smog_after"],
            "length_ratio": metrics["length_ratio"],
            "numbers_preserved": int(metrics["numbers_preserved"]),
        })
        print(f"FKGL {metrics['fkgl_before']:.2f} → {metrics['fkgl_after']:.2f}; "
              f"Length ratio {metrics['length_ratio']:.2f}; Numbers preserved: {bool(metrics['numbers_preserved'])}")

    # Aggregates
    n = len(results)
    if n:
        avg_fkgl_before = sum(r["fkgl_before"] for r in results if r["fkgl_before"] >= 0) / max(
            1, sum(1 for r in results if r["fkgl_before"] >= 0))
        avg_fkgl_after = sum(r["fkgl_after"] for r in results if r["fkgl_after"] >= 0) / max(
            1, sum(1 for r in results if r["fkgl_after"] >= 0))
        avg_len = sum(r["length_ratio"] for r in results) / n
        nums_ok = sum(r["numbers_preserved"] for r in results) / n
        print("\n=== AGGREGATES ===")
        print(f"Avg FKGL: {avg_fkgl_before:.2f} → {avg_fkgl_after:.2f}")
        print(f"Avg length ratio: {avg_len:.2f}")
        print(f"Numbers preserved: {nums_ok*100:.1f}%")

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write("idx,profile,fkgl_before,fkgl_after,smog_before,smog_after,length_ratio,numbers_preserved\n")
        for r in results:
            f.write(
                f"{r['idx']},{r['profile']},{r['fkgl_before']:.3f},{r['fkgl_after']:.3f},"
                f"{r['smog_before']:.3f},{r['smog_after']:.3f},{r['length_ratio']:.3f},{r['numbers_preserved']}\n"
            )
    print(f"Saved detailed metrics to {OUT_CSV}")


if __name__ == "__main__":
    main()
