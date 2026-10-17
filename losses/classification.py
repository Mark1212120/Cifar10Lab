from torch import nn


def make_loss(cfg, train):
    smoothing = cfg["loss"]["label_smoothing"] if train else cfg["loss"]["eval_label_smoothing"]
    return nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="mean")
