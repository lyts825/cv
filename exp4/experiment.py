"""MNIST 多层感知机实验：基础模型与参考改进模型。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import time

# 确定性 CUDA 矩阵乘法要求在首次创建 CUDA 上下文前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity


VERSION = "0.3.0"
ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {"baseline": "Baseline MLP", "improved": "Improved MLP"}


class BaselineMLP(nn.Module):
    """784 → 128 → 10，ReLU，与参考基础模型一致。"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x.flatten(1))))


class ImprovedMLP(nn.Module):
    """三个隐藏层，BatchNorm、GELU 和前两层后的 Dropout。"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.3)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.flatten(1)
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.act(self.bn2(self.fc2(x))))
        x = self.act(self.bn3(self.fc3(x)))
        return self.fc4(x)


def configure(seed: int, device: str = "auto") -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    return torch.device(device)


def load_data(data_dir: Path):
    """使用官方划分；一次性转换 uint8，等价于无增强的 ToTensor。"""
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        train = MNIST(data_dir, train=True, download=True)
        test = MNIST(data_dir, train=False, download=True)
    finally:
        socket.setdefaulttimeout(previous_timeout)
    if train.data.shape != (60000, 28, 28) or test.data.shape != (10000, 28, 28):
        raise ValueError("MNIST 数据形状与官方完整划分不符。")
    # 压缩文件由 torchvision 下载并校验；缓存再次使用时也检查完整性。
    fingerprints = {}
    for filename, md5 in MNIST.resources:
        archive = Path(train.raw_folder) / filename
        if not check_integrity(archive, md5):
            raise ValueError(f"MNIST 压缩文件缺失或 MD5 不符：{archive}")
        raw = archive.with_suffix("")
        fingerprints[raw.name] = hashlib.sha256(raw.read_bytes()).hexdigest()
    train_set = TensorDataset(train.data.unsqueeze(1).float().div_(255), train.targets)
    test_set = TensorDataset(test.data.unsqueeze(1).float().div_(255), test.targets)
    return train_set, test_set, fingerprints


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_samples(dataset, output_dir):
    images, labels = dataset.tensors
    fig, axes = plt.subplots(2, 5, figsize=(9, 4), constrained_layout=True)
    for digit, ax in enumerate(axes.flat):
        index = int(torch.where(labels == digit)[0][0])
        ax.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {digit} | index {index}")
        ax.axis("off")
    fig.suptitle("MNIST training samples: first occurrence of each class")
    save_figure(fig, output_dir / "samples.png")


def train_model(model, loader, optimizer, device, epochs, name):
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        start = time.perf_counter()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} 第 {epoch} 轮损失非有限值。")
            loss.backward()
            optimizer.step()
            # 按样本加权，末批不足 batch_size 时也有正确的权重。
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               "online_train_accuracy": correct / count,
               "seconds": time.perf_counter() - start}
        history.append(row)
        print(f"{name:8s} [{epoch:02d}/{epochs}] loss={row['train_loss']:.6f} "
              f"online_acc={row['online_train_accuracy']:.4%}", flush=True)
    return history


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, all_labels, all_predictions = 0.0, [], []
    for x, y in loader:
        logits = model(x.to(device))
        loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
        all_labels.append(y.numpy())
        all_predictions.append(logits.argmax(1).cpu().numpy())
    labels, predictions = np.concatenate(all_labels), np.concatenate(all_predictions)
    cm = np.bincount(10 * labels + predictions, minlength=100).reshape(10, 10)
    correct = int(np.trace(cm))
    if int(cm.sum()) != len(loader.dataset):
        raise RuntimeError("混淆矩阵计数与数据集样本数不一致。")
    per_class = []
    for digit in range(10):
        support = int(cm[digit].sum())
        true_positive = int(cm[digit, digit])
        predicted_count = int(cm[:, digit].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"digit": digit, "support": support, "correct": true_positive,
                          "incorrect": support - true_positive, "precision": precision,
                          "recall": recall, "f1": f1})
    return {"test_loss": loss_sum / len(labels), "test_accuracy": correct / len(labels),
            "correct": correct, "incorrect": len(labels) - correct,
            "macro_f1": float(np.mean([row["f1"] for row in per_class]))}, cm, per_class, labels, predictions


def plot_model(name, history, cm, metrics, test_set, labels, predictions, output_dir):
    title = MODEL_NAMES[name]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], "o-", color="#2563eb")
    axes[0].set(title="Training cross-entropy", ylabel="Loss per sample", xlabel="Epoch")
    axes[1].plot(epochs, [100 * r["online_train_accuracy"] for r in history], "o-", color="#15803d")
    axes[1].set(title="Online training accuracy (train mode)", ylabel="Accuracy (%)", xlabel="Epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(epochs)
    fig.suptitle(title)
    save_figure(fig, output_dir / "training_curves.png")

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "#172033")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
           title=f"{title} | test accuracy {metrics['test_accuracy']:.2%}")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Samples")
    save_figure(fig, output_dir / "confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    values = [metrics["correct"], metrics["incorrect"]]
    bars = axes[0].bar(["Correct", "Incorrect"], values, color=["#15803d", "#dc2626"])
    axes[0].bar_label(bars, padding=3)
    axes[0].set(title="Overall test predictions", ylabel="Samples", ylim=(0, 11000))
    correct = np.diag(cm)
    incorrect = cm.sum(axis=1) - correct
    x = np.arange(10)
    axes[1].bar(x - 0.2, correct, 0.4, label="Correct", color="#15803d")
    axes[1].bar(x + 0.2, incorrect, 0.4, label="Incorrect", color="#dc2626")
    axes[1].set(xticks=x, xlabel="True digit", ylabel="Samples", title="Per-class test predictions")
    axes[1].legend()
    fig.suptitle(title)
    save_figure(fig, output_dir / "prediction_statistics.png")

    images = test_set.tensors[0]
    for filename, indices, heading in [
        ("predictions.png", np.arange(12), "First 12 test samples"),
        ("errors.png", np.flatnonzero(labels != predictions)[:12], "First 12 misclassified test samples"),
    ]:
        fig, axes = plt.subplots(3, 4, figsize=(9, 7), constrained_layout=True)
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, indices):
            ax.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"#{index}: true {labels[index]} / pred {predictions[index]}", fontsize=10,
                         color="#15803d" if labels[index] == predictions[index] else "#dc2626")
        fig.suptitle(f"{title} | {heading}")
        save_figure(fig, output_dir / filename)


def refresh_comparison(output_dir: Path):
    """两组配置一致时才生成对比，防止混用不同运行的结果。"""
    paths = [output_dir / name / "summary.json" for name in MODEL_NAMES]
    if not all(path.is_file() for path in paths):
        return
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    keys = ("version", "seed", "epochs", "batch_size", "device", "environment", "dataset_sha256")
    if any(summaries[0][key] != summaries[1][key] for key in keys):
        raise ValueError("两组实验的公共配置不同，请用 --model all 重新运行后比较。")
    histories = []
    for name in MODEL_NAMES:
        with (output_dir / name / "history.csv").open(encoding="utf-8", newline="") as stream:
            histories.append(list(csv.DictReader(stream)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = ["#64748b", "#2563eb"]
    for summary, history, color in zip(summaries, histories, colors):
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in history], "o-", color=color,
                     label=MODEL_NAMES[summary["model"]])
    axes[0].set(title="Training cross-entropy (train mode)", xlabel="Epoch", ylabel="Loss per sample")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    values = [100 * s["metrics"]["test_accuracy"] for s in summaries]
    bars = axes[1].bar(list(MODEL_NAMES.values()), values, color=colors, width=0.55)
    axes[1].bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4)
    axes[1].set(title="Final test accuracy after 10 epochs" if summaries[0]["epochs"] == 10
                else f"Final test accuracy after {summaries[0]['epochs']} epochs",
                ylabel="Accuracy (%)", ylim=(0, 105))
    fig.suptitle("MNIST MLP comparison | fixed training budget")
    save_figure(fig, output_dir / "comparison.png")
    rows = [{"model": s["model"], "parameters": s["parameters"], **s["metrics"]} for s in summaries]
    save_csv(output_dir / "metrics.csv", rows)
    save_json(output_dir / "summary.json", {"version": VERSION, "experiments": summaries,
              "accuracy_gain_percentage_points": values[1] - values[0]})


def run_experiment(model_name="all", epochs=10, batch_size=2048, seed=2026,
                   device="auto", data_dir=None, output_dir=None):
    if model_name not in (*MODEL_NAMES, "all"):
        raise ValueError("model_name 必须为 baseline、improved 或 all。")
    if epochs < 1 or batch_size < 2 or (60000 % batch_size == 1):
        raise ValueError("epochs 必须为正数；训练批次至少含 2 个样本，以支持 BatchNorm。")
    data_dir = Path(data_dir) if data_dir else ROOT / ".local" / "data"
    output_dir = Path(output_dir) if output_dir else ROOT / "result"
    # 不保留上一次运行的汇总图，避免结果未更新时显示旧对比。
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("comparison.png", "metrics.csv", "summary.json"):
        (output_dir / filename).unlink(missing_ok=True)
    selected_device = configure(seed, device)
    train_set, test_set, fingerprints = load_data(data_dir)
    plot_samples(train_set, output_dir)
    results = []
    names = list(MODEL_NAMES) if model_name == "all" else [model_name]
    for name in names:
        configure(seed, str(selected_device))
        target = output_dir / name
        target.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, generator=generator)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        model = (BaselineMLP() if name == "baseline" else ImprovedMLP()).to(selected_device)
        optimizer = (torch.optim.SGD(model.parameters(), lr=0.1) if name == "baseline"
                     else torch.optim.Adam(model.parameters(), lr=1e-3))
        print(f"\n{name}: device={selected_device}, train={len(train_set)}, test={len(test_set)}\n{model}", flush=True)
        history = train_model(model, train_loader, optimizer, selected_device, epochs, name)
        metrics, cm, classes, labels, predictions = evaluate(model, test_loader, selected_device)
        save_csv(target / "history.csv", history)
        save_csv(target / "per_class_metrics.csv", classes)
        save_csv(target / "test_predictions.csv", [
            {"index": i, "true_label": int(y), "prediction": int(p)}
            for i, (y, p) in enumerate(zip(labels, predictions))])
        np.savetxt(target / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        # state_dict 只含参数及 BatchNorm 缓冲；转至 CPU 后可跨设备加载。
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, target / "model.pth")
        summary = {"version": VERSION, "model": name, "seed": seed, "epochs": epochs,
                   "batch_size": batch_size, "train_samples": len(train_set), "test_samples": len(test_set),
                   "device": str(selected_device),
                   "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                   "torchvision": torchvision.__version__, "numpy": np.__version__,
                                   "matplotlib": matplotlib.__version__, "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(selected_device)
                                   if selected_device.type == "cuda" else None},
                   "dataset_sha256": fingerprints,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "architecture": str(model),
                   "optimizer": {"name": type(optimizer).__name__, "learning_rate": optimizer.param_groups[0]["lr"]},
                   "metrics": metrics, "confusion_matrix": cm.tolist(),
                   "model_sha256": hashlib.sha256((target / "model.pth").read_bytes()).hexdigest()}
        plot_model(name, history, cm, metrics, test_set, labels, predictions, target)
        save_json(target / "summary.json", summary)
        results.append(summary)
        print(f"{name}: test_loss={metrics['test_loss']:.6f}, accuracy={metrics['test_accuracy']:.2%}, "
              f"correct={metrics['correct']}/10000", flush=True)
    refresh_comparison(output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODEL_NAMES], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".local" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    args = parser.parse_args()
    run_experiment(args.model, args.epochs, args.batch_size, args.seed, args.device, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
