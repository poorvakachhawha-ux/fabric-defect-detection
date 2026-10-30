"""Compare Roboflow and PyTorch on the shared 50 images and collect failure cases."""
import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

from common import split_dir


def load(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}; run the script that produces it first")
    with path.open() as f:
        return {row["image_name"]: row for row in csv.DictReader(f)}


def summarize(rows: dict[str, dict]) -> tuple[float, float]:
    accuracy = sum(int(r["correct"]) for r in rows.values()) / len(rows)
    latency = statistics.mean(float(r["latency_ms"]) for r in rows.values())
    return accuracy, latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--roboflow", type=Path, default=Path("results/roboflow/predictions_detailed.csv"))
    parser.add_argument("--pytorch", type=Path, default=Path("results/pytorch/predictions_pytorch.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("results/pytorch/metrics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/failure_cases"))
    args = parser.parse_args()

    roboflow, pytorch = load(args.roboflow), load(args.pytorch)
    if roboflow.keys() != pytorch.keys():
        raise SystemExit("The two prediction files cover different images; rerun both on eval_images.txt")

    rf_acc, rf_latency = summarize(roboflow)
    pt_acc, pt_latency = summarize(pytorch)
    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}

    print(f"{'':28}{'Roboflow':>12}{'PyTorch':>12}")
    print(f"{'Accuracy on 50 images':28}{rf_acc:>12.1%}{pt_acc:>12.1%}")
    print(f"{'Mean latency (ms)':28}{rf_latency:>12.1f}{pt_latency:>12.1f}")
    if metrics:
        print(f"{'Full test accuracy':28}{0.9638:>12.1%}{metrics['test_accuracy']:>12.1%}")
        print(f"{'Training time (min)':28}{20:>12.1f}{metrics['train_seconds'] / 60:>12.1f}")

    test_dir = split_dir(args.data_dir, "test")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in sorted(roboflow):
        rf, pt = roboflow[name], pytorch[name]
        if rf["correct"] == "1" and pt["correct"] == "1":
            continue
        failures.append({
            "image_name": name,
            "true_label": rf["true_label"],
            "roboflow_pred": rf["predicted_label"],
            "roboflow_conf": rf["confidence"],
            "pytorch_pred": pt["predicted_label"],
            "pytorch_conf": pt["confidence"],
            "wrong_in": "both" if rf["correct"] == pt["correct"] == "0"
            else ("roboflow" if rf["correct"] == "0" else "pytorch"),
        })
        source = test_dir / rf["true_label"] / name
        if source.exists():
            shutil.copy2(source, args.out_dir / name)

    with (args.out_dir / "failure_cases.csv").open("w", newline="") as f:
        fieldnames = ["image_name", "true_label", "roboflow_pred", "roboflow_conf",
                      "pytorch_pred", "pytorch_conf", "wrong_in"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)

    print(f"\n{len(failures)} images wrong in at least one model; copied to {args.out_dir}/")
    for row in failures:
        print(
            f"  {row['image_name']}: true {row['true_label']}, "
            f"Roboflow {row['roboflow_pred']} ({float(row['roboflow_conf']):.2f}), "
            f"PyTorch {row['pytorch_pred']} ({float(row['pytorch_conf']):.2f})"
        )


if __name__ == "__main__":
    main()
