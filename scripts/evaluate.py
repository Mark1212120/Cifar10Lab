import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from datasets.cifar10 import make_test
from evaluation.metrics import evaluate, plot_confusion
from losses.classification import make_loss
from models.resnet18 import build_model
from tools.runtime import load_config, prepare_runtime, project_path, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_resnet18.yaml")
    args = parser.parse_args()
    requested = load_config(args.config)
    out = project_path(requested["experiment"]["output_dir"])
    result_file = out / requested["logging"]["result_file"]
    if result_file.exists():
        raise FileExistsError("Final test results already exist; read results.json instead of testing again")
    last_file = out / requested["checkpoint"]["last_name"]
    last = torch.load(last_file, map_location="cpu", weights_only=False)
    if last["epoch"] < last["config"]["training"]["epochs"]:
        raise RuntimeError("Complete the planned training epochs before final testing")
    del last
    checkpoint = torch.load(out / requested["checkpoint"]["best_name"], map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if project_path(cfg["experiment"]["output_dir"]) != out:
        raise ValueError("Checkpoint does not belong to this experiment directory")
    split_file = project_path(cfg["paths"]["split_indices"])
    if file_hash(split_file) != checkpoint["split_hash"]:
        raise ValueError("Saved training split changed")
    device = prepare_runtime(cfg)
    model = build_model(cfg, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    classes = cfg["dataset"]["class_names"]
    metrics, predictions = evaluate(model, make_test(cfg), device, make_loss(cfg, False),
                                    cfg["runtime"]["amp_enabled"], classes, save_predictions=True)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_index", "true_label", "pred_label", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    plot_confusion(metrics["confusion_matrix"], classes, out / "confusion_matrix.png")
    result = dict(experiment=cfg["experiment"]["name"], seed=cfg["experiment"]["seed"],
                  best_epoch=checkpoint["epoch"], best_val_accuracy=checkpoint["val_accuracy"],
                  pretraining="ImageNet-1K / ResNet18 IMAGENET1K_V1", test_size=len(predictions),
                  split_sha256=checkpoint["split_hash"], classes=classes, metrics=metrics, config=cfg)
    source_file = project_path(cfg["paths"]["data_root"]) / "data_source.json"
    if source_file.exists():
        result["data_source"] = json.loads(source_file.read_text(encoding="utf-8"))
    temp = result_file.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(result_file)
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Test accuracy: {100 * metrics['accuracy']:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Results: {result_file}")


if __name__ == "__main__":
    main()
