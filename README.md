# Fabric defect detection: Roboflow vs PyTorch

Single-label classification of fabric images into `defect_free`, `defective` and `waste`.
A Roboflow-hosted ResNet-18 (`sanya-mohanty/poorva-fabric-defect-detection-idm-2-1-resnet18-t1`)
is benchmarked against a ResNet-18 fine-tuned locally in PyTorch on the same 70/20/10 split.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ROBOFLOW_API_KEY=...   # Roboflow workspace settings -> API keys
```

## Run

```bash
python download_dataset.py   # dataset version 1 in folder format -> dataset/
python infer.py              # Roboflow API on 50 test images -> predictions.csv
python train_pytorch.py      # ResNet-18 fine-tune -> results/pytorch/
python compare.py            # side-by-side numbers and failure cases -> results/failure_cases/
python report.py             # 3-page comparison -> report.pdf
```

The 50 benchmark images are chosen once, balanced across classes, and pinned in
`eval_images.txt` so both models are timed and scored on the same images.

## Outputs

| File | Contents |
|---|---|
| `predictions.csv` | Roboflow predictions: image name, predicted label, confidence |
| `results/roboflow/` | Same predictions with ground truth and per-call latency, plus a summary |
| `results/pytorch/metrics.json` | Training time, test accuracy, per-class metrics, confusion matrix, latency |
| `results/pytorch/test_misclassified.csv` | Every test image the PyTorch model got wrong |
| `results/failure_cases/` | Images either model got wrong on the 50, for the failure analysis |

Roboflow latency includes the network round trip to the hosted API. PyTorch latency is local
(image load, preprocessing and forward pass); `forward_ms` isolates the model alone.
