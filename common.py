from __future__ import annotations

import random
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_ID = "sanya-mohanty/poorva-fabric-defect-detection-idm-2-1-resnet18-t1"
WORKSPACE = "sanya-mohanty"
PROJECT = "poorva-fabric-defect-detection-idm-2"
VERSION = 1


def split_dir(data_dir: Path, split: str) -> Path:
    candidates = {"valid": ("valid", "val", "validation")}.get(split, (split,))
    for name in candidates:
        path = data_dir / name
        if path.is_dir():
            return path
    raise SystemExit(f"Could not find the '{split}' split under {data_dir}")


def list_split(directory: Path) -> list[tuple[Path, str]]:
    items = []
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() in IMAGE_EXTS:
                items.append((image, class_dir.name))
    if not items:
        raise SystemExit(f"No images found in class folders under {directory}")
    return items


def select_eval_images(
    test_dir: Path, n: int, manifest: Path, seed: int = 42
) -> list[tuple[Path, str]]:
    """Pick n test images balanced across classes, pinned in a manifest so every
    script benchmarks exactly the same images."""
    if manifest.exists():
        selected = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            path = test_dir / line.strip()
            if not path.exists():
                raise SystemExit(f"{manifest} lists {line!r}, which is missing from {test_dir}")
            selected.append((path, path.parent.name))
        if len(selected) != n:
            raise SystemExit(
                f"{manifest} has {len(selected)} images but {n} were requested; "
                "delete it to reselect"
            )
        return selected

    by_class: dict[str, list[Path]] = {}
    for path, label in list_split(test_dir):
        by_class.setdefault(label, []).append(path)
    rng = random.Random(seed)
    for paths in by_class.values():
        rng.shuffle(paths)

    selected = []
    while len(selected) < n and any(by_class.values()):
        for label in sorted(by_class):
            if by_class[label] and len(selected) < n:
                selected.append((by_class[label].pop(), label))
    if len(selected) < n:
        raise SystemExit(f"Only {len(selected)} test images available, need {n}")

    manifest.write_text(
        "\n".join(f"{label}/{path.name}" for path, label in selected) + "\n"
    )
    return selected
