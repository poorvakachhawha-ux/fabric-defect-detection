"""Fine-tune an ImageNet ResNet-18 on the same Roboflow split and benchmark it."""
import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common import select_eval_images, split_dir

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_transforms(size: int, augment: bool):
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_steps = [transforms.Resize((size, size))]
    if augment:
        train_steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15),
        ]
    train_steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(train_steps), eval_tf


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += inputs.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred, confidences = [], [], []
    for inputs, targets in loader:
        probs = torch.softmax(model(inputs.to(device)), dim=1).cpu()
        conf, pred = probs.max(1)
        y_true += targets.tolist()
        y_pred += pred.tolist()
        confidences += conf.tolist()
    return y_true, y_pred, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/pytorch"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--manifest", type=Path, default=Path("eval_images.txt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_tf, eval_tf = build_transforms(args.img_size, augment=not args.no_augment)
    train_ds = datasets.ImageFolder(split_dir(args.data_dir, "train"), train_tf)
    valid_ds = datasets.ImageFolder(split_dir(args.data_dir, "valid"), eval_tf)
    test_ds = datasets.ImageFolder(split_dir(args.data_dir, "test"), eval_tf)
    if not (train_ds.classes == valid_ds.classes == test_ds.classes):
        raise SystemExit("train, valid and test have different class folders")
    classes = train_ds.classes
    print(f"Classes: {classes}")
    print(f"Images: train {len(train_ds)}, valid {len(valid_ds)}, test {len(test_ds)}")

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.workers}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_acc, best_epoch, best_state, stale = [], -1.0, 0, None, 0
    sync(device)
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion, device)
        scheduler.step()
        sync(device)
        epoch_seconds = time.perf_counter() - epoch_started
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "valid_loss": round(valid_loss, 4),
            "valid_acc": round(valid_acc, 4),
            "seconds": round(epoch_seconds, 1),
        })
        print(
            f"epoch {epoch:2d}  train {train_loss:.3f}/{train_acc:.3f}  "
            f"valid {valid_loss:.3f}/{valid_acc:.3f}  {epoch_seconds:.0f}s"
        )
        if valid_acc > best_acc:
            best_acc, best_epoch, stale = valid_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no validation gain for {args.patience} epochs")
                break
    sync(device)
    train_seconds = time.perf_counter() - train_started

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes}, args.out_dir / "best_resnet18.pt")

    y_true, y_pred, confidences = predict(model, test_loader, device)
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    print("\n" + classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    with (args.out_dir / "test_misclassified.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_label", "confidence"])
        for (path, _), t, p, c in zip(test_ds.samples, y_true, y_pred, confidences):
            if t != p:
                writer.writerow([path, classes[t], classes[p], f"{c:.6f}"])

    with (args.out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    eval_images = select_eval_images(split_dir(args.data_dir, "test"), args.n_eval, args.manifest)
    model.eval()
    with torch.no_grad():
        warmup = eval_tf(Image.open(eval_images[0][0]).convert("RGB")).unsqueeze(0).to(device)
        for _ in range(10):
            model(warmup)
        sync(device)

        rows = []
        for path, true_label in eval_images:
            started = time.perf_counter()
            tensor = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            sync(device)
            forward_started = time.perf_counter()
            probs = torch.softmax(model(tensor), dim=1)
            sync(device)
            finished = time.perf_counter()
            conf, pred = probs.max(1)
            label = classes[pred.item()]
            rows.append({
                "image_name": path.name,
                "true_label": true_label,
                "predicted_label": label,
                "confidence": f"{conf.item():.6f}",
                "correct": int(label == true_label),
                "latency_ms": f"{(finished - started) * 1000:.2f}",
                "forward_ms": f"{(finished - forward_started) * 1000:.2f}",
            })

    with (args.out_dir / "predictions_pytorch.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latencies = [float(r["latency_ms"]) for r in rows]
    forwards = [float(r["forward_ms"]) for r in rows]
    metrics = {
        "model": "resnet18 (ImageNet weights, full fine-tune)",
        "device": str(device),
        "classes": classes,
        "images": {"train": len(train_ds), "valid": len(valid_ds), "test": len(test_ds)},
        "hyperparameters": {
            "epochs_max": args.epochs,
            "epochs_run": len(history),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "img_size": args.img_size,
            "augment": not args.no_augment,
        },
        "train_seconds": round(train_seconds, 1),
        "best_epoch": best_epoch,
        "best_valid_acc": round(best_acc, 4),
        "test_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "test_macro_precision": round(report["macro avg"]["precision"], 4),
        "test_macro_recall": round(report["macro avg"]["recall"], 4),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        "per_class": {
            name: {k: round(report[name][k], 4) for k in ("precision", "recall", "f1-score", "support")}
            for name in classes
        },
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": matrix.tolist()},
        "eval_50": {
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "latency_ms_mean": statistics.mean(latencies),
            "latency_ms_median": statistics.median(latencies),
            "forward_ms_mean": statistics.mean(forwards),
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Training time: {train_seconds / 60:.1f} min ({len(history)} epochs, best epoch {best_epoch})")
    print(f"Test accuracy: {metrics['test_accuracy']:.1%}")
    print(
        f"Latency on {len(rows)} images: mean {metrics['eval_50']['latency_ms_mean']:.1f} ms "
        f"(forward only {metrics['eval_50']['forward_ms_mean']:.1f} ms)"
    )
    print(f"Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
