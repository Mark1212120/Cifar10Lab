import torch
from torchvision.models import resnet18, ResNet18_Weights


def build_model(cfg, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, cfg["model"]["num_classes"])
    if sum(p.numel() for p in model.parameters()) != 11181642:
        raise ValueError("Unexpected ResNet-18 parameter count")
    return model


def set_train_mode(model, cfg):
    model.train()
    if cfg["model"]["freeze_batchnorm_running_stats"]:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                if not cfg["model"]["train_batchnorm_affine"]:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
