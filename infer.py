"""Run the Roboflow-hosted classifier on 50 unseen test images and write predictions.csv."""
import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

from inference_sdk import InferenceHTTPClient

from common import MODEL_ID, select_eval_images, split_dir


def top_prediction(result) -> tuple[str, float]:
    if isinstance(result, list):
        result = result[0]
    if "top" in result:
        return result["top"], float(result["confidence"])
    predictions = result.get("predictions", [])
    if isinstance(predictions, dict):
        label, info = max(predictions.items(), key=lambda kv: kv[1]["confidence"])
        return label, float(info["confidence"])
    best = max(predictions, key=lambda p: p["confidence"])
    return best["class"], float(best["confidence"])


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--api-url", default="https://serverless.roboflow.com")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--manifest", type=Path, default=Path("eval_images.txt"))
    parser.add_argument("--out", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/roboflow"))
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY first")

    images = select_eval_images(split_dir(args.data_dir, "test"), args.n, args.manifest)
    client = InferenceHTTPClient(api_url=args.api_url, api_key=api_key)

    client.infer(str(images[0][0]), model_id=args.model_id)  # warmup, not timed

    rows = []
    for path, true_label in images:
        started = time.perf_counter()
        result = client.infer(str(path), model_id=args.model_id)
        latency_ms = (time.perf_counter() - started) * 1000
        label, confidence = top_prediction(result)
        rows.append({
            "image_name": path.name,
            "true_label": true_label,
            "predicted_label": label,
            "confidence": f"{confidence:.6f}",
            "correct": int(label == true_label),
            "latency_ms": f"{latency_ms:.2f}",
        })
        print(f"{path.name}: {label} ({confidence:.3f}) in {latency_ms:.0f} ms")

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "predictions_detailed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latencies = [float(row["latency_ms"]) for row in rows]
    summary = {
        "model_id": args.model_id,
        "images": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": percentile(latencies, 95),
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {args.out} ({len(rows)} rows)")
    print(f"Accuracy on these {len(rows)} images: {summary['accuracy']:.1%}")
    print(
        f"Latency: mean {summary['latency_ms_mean']:.0f} ms, "
        f"median {summary['latency_ms_median']:.0f} ms, p95 {summary['latency_ms_p95']:.0f} ms"
    )


if __name__ == "__main__":
    main()
