import csv

import numpy as np
import torch


@torch.inference_mode()
def evaluate(model, loader, device, criterion, amp, classes, save_predictions=False):
    model.eval()
    size = len(classes)
    cm = np.zeros((size, size), dtype=np.int64)
    loss_sum, seen, predictions = 0.0, 0, []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits.float(), labels)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite evaluation loss")
        predicted = logits.argmax(1)
        true_cpu, pred_cpu = labels.cpu().numpy(), predicted.cpu().numpy()
        np.add.at(cm, (true_cpu, pred_cpu), 1)
        if save_predictions:
            confidence = logits.float().softmax(1).amax(1).cpu().tolist()
            for offset, (true_id, pred_id, conf) in enumerate(zip(true_cpu, pred_cpu, confidence)):
                predictions.append(dict(test_index=seen + offset, true_label=int(true_id),
                                        pred_label=int(pred_id), confidence=float(conf)))
        seen += labels.numel()
        loss_sum += loss.item() * labels.numel()
    if seen != len(loader.dataset):
        raise ValueError("Evaluation missed samples")
    tp = np.diag(cm).astype(float)
    actual, predicted_counts = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted_counts, out=np.zeros(size), where=predicted_counts != 0)
    recall = np.divide(tp, actual, out=np.zeros(size), where=actual != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(size), where=denom != 0)
    per_class = [dict(class_name=name, accuracy=float(recall[i]), precision=float(precision[i]),
                      recall=float(recall[i]), f1=float(f1[i]), support=int(actual[i]))
                 for i, name in enumerate(classes)]
    metrics = dict(accuracy=float(tp.sum() / seen), loss=float(loss_sum / seen),
                   macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                   macro_f1=float(f1.mean()), per_class=per_class, confusion_matrix=cm.tolist())
    return metrics, predictions


def plot_curves(csv_file, output, smoothing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with csv_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label=f"Train CE (LS={smoothing})")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="Validation CE (LS=0)")
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, [100 * float(row["train_acc"]) for row in rows], label="Train")
    axes[1].plot(epochs, [100 * float(row["val_acc"]) for row in rows], label="Validation")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_confusion(matrix, classes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(10, 8))
    picture = axis.imshow(cm, cmap="Blues")
    fig.colorbar(picture, ax=axis)
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    for row in range(len(classes)):
        for col in range(len(classes)):
            axis.text(col, row, str(cm[row, col]), ha="center", va="center",
                       color="white" if cm[row, col] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
