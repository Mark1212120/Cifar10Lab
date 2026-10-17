import argparse
import itertools
import logging
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from datasets.cifar10 import make_train_val
from evaluation.metrics import evaluate, plot_curves
from losses.classification import make_loss
from models.resnet18 import build_model, set_train_mode
from tools.runtime import (load_config, prepare_runtime, project_path, capture_rng, restore_rng,
                           atomic_save, setup_log, save_environment, append_metrics, trim_metrics)


def make_optimizer(model, cfg):
    settings = cfg["optimizer"]
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = settings["classifier_lr"] if name.startswith("fc.") else settings["backbone_lr"]
        no_decay = settings["no_decay_for_bias_and_norm"] and (name.endswith(".bias") or parameter.ndim == 1)
        decay = 0.0 if no_decay else settings["weight_decay"]
        groups.setdefault((lr, decay), []).append(parameter)
    params = [dict(params=values, lr=lr, base_lr=lr, weight_decay=decay)
              for (lr, decay), values in groups.items()]
    return torch.optim.AdamW(params, betas=tuple(settings["betas"]), eps=settings["eps"])


def set_lr(optimizer, successful_updates, cfg):
    schedule = cfg["scheduler"]
    u = successful_updates + 1
    warmup, total = schedule["expected_warmup_updates"], schedule["expected_total_updates"]
    if u <= warmup:
        start = schedule["warmup_start_factor"]
        factor = start + (1 - start) * (u - 1) / max(1, warmup - 1)
    else:
        progress = min(1.0, (u - warmup) / (total - warmup))
        minimum = schedule["min_lr_ratio"]
        factor = minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def train_epoch(model, loader, device, optimizer, scaler, criterion, cfg, updates, limit=None):
    set_train_mode(model, cfg)
    accumulation = cfg["training"]["gradient_accumulation_steps"]
    iterator = iter(loader) if limit is None else itertools.islice(iter(loader), limit)
    loss_sum, correct, seen, micro_count, skips = 0.0, 0, 0, 0, 0
    amp = cfg["runtime"]["amp_enabled"]
    while True:
        group = list(itertools.islice(iterator, accumulation))
        if not group:
            break
        actual_samples = sum(labels.numel() for _, labels in group)
        optimizer.zero_grad(set_to_none=True)
        set_lr(optimizer, updates, cfg)
        for images, labels in group:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if micro_count == 0 and tuple(images.shape[1:]) != (3, cfg["transforms"]["input_size"], cfg["transforms"]["input_size"]):
                raise ValueError("Unexpected input shape")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(images)
                if logits.shape != (labels.numel(), cfg["dataset"]["num_classes"]):
                    raise ValueError("Unexpected logits shape")
                loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss * labels.numel() / actual_samples).backward()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.numel()
            micro_count += 1
            if micro_count % cfg["logging"]["console_every_micro_batches"] == 0:
                logging.info("micro_batch=%d loss=%.4f train_acc=%.2f%%", micro_count,
                             loss_sum / seen, 100 * correct / seen)
        if model.fc.weight.grad is None or model.conv1.weight.grad is None:
            raise RuntimeError("Expected classifier and backbone gradients")
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_max_norm"])
        if not scaler.is_enabled() and not torch.isfinite(norm).item():
            raise FloatingPointError("Non-finite gradients")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        if skipped:
            skips += 1
            logging.warning("AMP skipped an optimizer update")
            if skips >= 10:
                raise FloatingPointError("Too many AMP skips; inspect the run instead of continuing")
        else:
            updates += 1
    return dict(loss=loss_sum / seen, accuracy=correct / seen, skips=skips), updates


def smoke_check(model, loader, generator, device, cfg):
    count = cfg["runtime"]["smoke_test_micro_batches"]
    if not count:
        return
    logging.info("Preflight: %d micro-batches; weights and random states will be reset", count)
    initial = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    rng = capture_rng(generator)
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["runtime"]["amp_enabled"])
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    train_epoch(model, loader, device, optimizer, scaler, make_loss(cfg, True), cfg, 0, limit=count)
    torch.cuda.synchronize(device)
    for name, tensor in model.state_dict().items():
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            if not torch.equal(tensor.cpu(), initial[name]):
                raise RuntimeError("BatchNorm running statistics changed")
    logging.info("Preflight passed: seconds=%.1f peak_allocated=%.1f MiB peak_reserved=%.1f MiB",
                 time.perf_counter() - start, torch.cuda.max_memory_allocated(device) / 2**20,
                 torch.cuda.max_memory_reserved(device) / 2**20)
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    restore_rng(rng, generator)
    del optimizer, scaler, initial


def save_best(state, path):
    best = dict(model=state["best_model"], config=state["config"], epoch=state["best_epoch"],
                val_accuracy=state["best_accuracy"], val_loss=state["best_loss"],
                split_hash=state["split_hash"])
    atomic_save(best, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_resnet18.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = prepare_runtime(cfg)
    out = project_path(cfg["experiment"]["output_dir"])
    last_path, best_path = out / cfg["checkpoint"]["last_name"], out / cfg["checkpoint"]["best_name"]
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError("No last.pth; cannot resume before the first complete epoch")
    elif out.exists():
        raise FileExistsError(f"Run directory already exists: {out}. Use --resume for an existing run.")
    out.mkdir(parents=True, exist_ok=True)
    setup_log(out, args.resume)
    logging.info("Project=%s GPU=%s resume=%s", ROOT, torch.cuda.get_device_name(device), args.resume)
    if not args.resume:
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        save_environment(out)
    train_loader, val_loader, generator, split_hash = make_train_val(cfg, must_exist=args.resume)
    model = build_model(cfg, pretrained=not args.resume).to(device)
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["runtime"]["amp_enabled"])
    start_epoch, updates, best_accuracy, best_loss, best_epoch, best_model = 0, 0, -1.0, float("inf"), 0, None
    metrics_file = out / cfg["logging"]["epoch_metrics_file"]
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config"] != cfg or state["split_hash"] != split_hash:
            raise ValueError("Resume configuration or split differs from original run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        updates = state["scheduler"]["successful_updates"]
        best_accuracy, best_loss, best_epoch = state["best_accuracy"], state["best_loss"], state["best_epoch"]
        best_model = state["best_model"]
        restore_rng(state["rng"], generator)
        trim_metrics(metrics_file, start_epoch)
        save_best(state, best_path)
        logging.info("Resume after epoch=%d, successful_updates=%d", start_epoch, updates)
        del state
    else:
        smoke_check(model, train_loader, generator, device, cfg)
        optimizer = make_optimizer(model, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["runtime"]["amp_enabled"])
    for epoch in range(start_epoch + 1, cfg["training"]["epochs"] + 1):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        train_metrics, updates = train_epoch(model, train_loader, device, optimizer, scaler,
                                             make_loss(cfg, True), cfg, updates)
        val, _ = evaluate(model, val_loader, device, make_loss(cfg, False),
                           cfg["runtime"]["amp_enabled"], cfg["dataset"]["class_names"])
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - begin
        last_lr = {group["base_lr"]: group["lr"] for group in optimizer.param_groups}
        row = dict(epoch=epoch, lr_backbone=last_lr[cfg["optimizer"]["backbone_lr"]],
                   lr_classifier=last_lr[cfg["optimizer"]["classifier_lr"]],
                   train_loss=train_metrics["loss"], train_acc=train_metrics["accuracy"],
                   val_loss=val["loss"], val_acc=val["accuracy"], epoch_seconds=seconds,
                   peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                   amp_skips=train_metrics["skips"])
        append_metrics(metrics_file, row)
        improved = val["accuracy"] > best_accuracy or (val["accuracy"] == best_accuracy and val["loss"] < best_loss)
        if improved:
            best_accuracy, best_loss, best_epoch = val["accuracy"], val["loss"], epoch
            best_model = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(), scheduler=dict(successful_updates=updates),
                     config=cfg, split_hash=split_hash, rng=capture_rng(generator),
                     best_accuracy=best_accuracy, best_loss=best_loss, best_epoch=best_epoch,
                     best_model=best_model)
        atomic_save(state, last_path)
        if improved:
            save_best(state, best_path)
        plot_curves(metrics_file, out / "training_curves.png", cfg["loss"]["label_smoothing"])
        logging.info("epoch=%d/%d train_acc=%.2f%% val_acc=%.2f%% seconds=%.1f best_epoch=%d",
                     epoch, cfg["training"]["epochs"], 100 * train_metrics["accuracy"],
                     100 * val["accuracy"], seconds, best_epoch)
    plot_curves(metrics_file, out / "training_curves.png", cfg["loss"]["label_smoothing"])
    logging.info("Training complete. Run: python scripts/evaluate.py --config %s", args.config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted. Resume from last completed epoch using --resume.")
