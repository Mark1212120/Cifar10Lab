import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from torchvision import transforms as T

from tools.runtime import file_hash, project_path


class CIFAR10(Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        root = __import__('pathlib').Path(root)
        split = 'train' if train else 'test'
        cache = root / f'cifar10_{split}.npz'
        self.transform = transform
        self.fallback = None
        if not cache.exists():
            self.fallback = TorchCIFAR10(root, train=train, download=download, transform=transform)
            self.targets, self.classes = self.fallback.targets, self.fallback.classes
            return
        manifest = json.loads((root / 'data_source.json').read_text(encoding='utf-8'))
        if file_hash(cache) != manifest[split]['npz_sha256']:
            raise ValueError('Local converted data checksum mismatch')
        with np.load(cache, allow_pickle=False) as data:
            self.images = data['images']
            self.targets = data['labels'].tolist()
        self.classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        expected = 50000 if train else 10000
        if self.images.shape != (expected,32,32,3) or self.images.dtype != np.uint8:
            raise ValueError('Invalid CIFAR-10 image shape or dtype')
        if not np.all(np.bincount(self.targets,minlength=10) == expected//10):
            raise ValueError('Invalid CIFAR-10 class counts')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.fallback is not None:
            return self.fallback[index]
        image = Image.fromarray(self.images[index])
        return (self.transform(image) if self.transform else image), self.targets[index]


def make_transform(cfg, train):
    t = cfg["transforms"]
    items = []
    if train:
        items += [T.RandomCrop(t["train_random_crop_size"], padding=t["train_random_crop_padding"]),
                  T.RandomHorizontalFlip(t["train_horizontal_flip_probability"])]
    items += [T.Resize((t["input_size"], t["input_size"]),
                       interpolation=T.InterpolationMode.BILINEAR, antialias=t["resize_antialias"]),
              T.ToTensor(), T.Normalize(t["mean"], t["std"])]
    return T.Compose(items)


def make_train_val(cfg, must_exist=False):
    data_root = project_path(cfg["paths"]["data_root"])
    ds = cfg["dataset"]
    train_base = CIFAR10(data_root, train=True, download=ds["download"],
                         transform=make_transform(cfg, True))
    val_base = CIFAR10(data_root, train=True, download=False, transform=make_transform(cfg, False))
    if train_base.classes != ds["class_names"]:
        raise ValueError("CIFAR-10 class order mismatch")
    labels = np.asarray(train_base.targets)
    split_file = project_path(cfg["paths"]["split_indices"])
    if must_exist and not split_file.exists():
        raise FileNotFoundError("Resume requires the original saved split indices")
    if not split_file.exists():
        rng = np.random.default_rng(ds["split_seed"])
        train_ids, val_ids = [], []
        for class_id in range(ds["num_classes"]):
            ids = np.flatnonzero(labels == class_id)
            rng.shuffle(ids)
            count = ds["val_per_class"]
            val_ids.extend(ids[:count].tolist())
            train_ids.extend(ids[count:].tolist())
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(dict(seed=ds["split_seed"], train=train_ids, val=val_ids)),
                              encoding="utf-8")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    train_ids, val_ids = split["train"], split["val"]
    if split["seed"] != ds["split_seed"]:
        raise ValueError("Split seed mismatch")
    if len(train_ids) != ds["train_size"] or len(val_ids) != ds["val_size"]:
        raise ValueError("Split size mismatch")
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate split indices")
    if set(train_ids) & set(val_ids) or set(train_ids + val_ids) != set(range(len(labels))):
        raise ValueError("Train/validation split overlap or missing samples")
    for ids, count in ((train_ids, ds["train_per_class"]), (val_ids, ds["val_per_class"])):
        if not np.all(np.bincount(labels[ids], minlength=ds["num_classes"]) == count):
            raise ValueError("Split is not class balanced")
    generator = torch.Generator().manual_seed(cfg["experiment"]["seed"])
    common = dict(num_workers=0, pin_memory=cfg["runtime"]["pin_memory"])
    train_loader = DataLoader(Subset(train_base, train_ids), batch_size=cfg["training"]["batch_size"],
                              shuffle=True, drop_last=False, generator=generator, **common)
    val_loader = DataLoader(Subset(val_base, val_ids), batch_size=cfg["evaluation"]["batch_size"],
                            shuffle=False, **common)
    return train_loader, val_loader, generator, file_hash(split_file)


def make_test(cfg):
    dataset = CIFAR10(project_path(cfg["paths"]["data_root"]), train=False,
                      download=cfg["dataset"]["download"], transform=make_transform(cfg, False))
    if len(dataset) != cfg["dataset"]["test_size"] or dataset.classes != cfg["dataset"]["class_names"]:
        raise ValueError("Test data size or class order mismatch")
    return DataLoader(dataset, batch_size=cfg["evaluation"]["batch_size"], shuffle=False,
                      num_workers=0, pin_memory=cfg["runtime"]["pin_memory"])
