import csv
import hashlib
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["epoch", "lr_backbone", "lr_classifier", "train_loss", "train_acc",
          "val_loss", "val_acc", "epoch_seconds", "peak_allocated_mib",
          "peak_reserved_mib", "amp_skips"]


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must stay inside project: {path}")
    return path


def load_config(value):
    with project_path(value).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    train = cfg["training"]
    batch = train["batch_size"]
    accum = train["gradient_accumulation_steps"]
    if not (1 <= batch <= min(4, train["max_physical_batch_size"])):
        raise ValueError("Physical batch must be between 1 and 4")
    if accum < 1 or batch * accum != train["effective_batch_size"]:
        raise ValueError("Check batch, accumulation and effective_batch_size")
    if not (0 < cfg["evaluation"]["batch_size"] <= 4):
        raise ValueError("Evaluation batch must be between 1 and 4")
    if not 0 < cfg["scheduler"]["warmup_epochs"] < train["epochs"]:
        raise ValueError("Require 0 < warmup_epochs < epochs")
    if cfg["model"]["name"] != "torchvision_resnet18":
        raise ValueError("This implementation supports ResNet-18 only")
    if cfg["runtime"]["num_workers"] != 0:
        raise ValueError("Keep num_workers=0 for this first experiment")
    if cfg["runtime"]["amp_dtype"] != "float16":
        raise ValueError("This implementation uses float16 AMP")
    batches = math.ceil(cfg["dataset"]["train_size"] / batch)
    updates = math.ceil(batches / accum)
    cfg["scheduler"].update(expected_updates_per_epoch=updates,
                            expected_total_updates=updates * train["epochs"],
                            expected_warmup_updates=updates * cfg["scheduler"]["warmup_epochs"])
    return cfg


def prepare_runtime(cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch_cache = project_path(cfg["paths"]["torch_cache"])
    mpl_cache = project_path(cfg["paths"]["matplotlib_cache"])
    for path in (torch_cache, mpl_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Fix the environment before training.")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.backends.cudnn.benchmark = cfg["runtime"]["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = cfg["runtime"]["cudnn_deterministic"]
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed = cfg["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device(cfg["runtime"]["device"])


def capture_rng(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(state, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def setup_log(out, resume):
    logging.captureWarnings(True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / "run.log", mode="a" if resume else "w",
                                                      encoding="utf-8")], force=True)


def save_environment(out):
    import torchvision
    lines = [sys.version, sys.executable, f"torch={torch.__version__}",
             f"torchvision={torchvision.__version__}", f"CUDA={torch.version.cuda}",
             f"cuDNN={torch.backends.cudnn.version()}",
             f"GPU={torch.cuda.get_device_name(0)}"]
    for command in (["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                    [sys.executable, "-m", "pip", "freeze"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        lines.append(result.stdout or result.stderr)
    (out / "environment.txt").write_text("\n".join(lines), encoding="utf-8")


def append_metrics(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def trim_metrics(path, completed_epoch):
    if not path.exists():
        raise FileNotFoundError("Resume needs the original metrics.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) <= completed_epoch]
    if [int(row["epoch"]) for row in rows] != list(range(1, completed_epoch + 1)):
        raise ValueError("metrics.csv does not match the checkpoint epoch")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
