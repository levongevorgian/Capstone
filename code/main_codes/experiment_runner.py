#!/usr/bin/env python3
"""Experiment pipeline for label-free drift detection in image classification.

This module is designed to fix the methodological gaps from the original
notebook:
- isolated drift families instead of mixed corruptions
- explicit accuracy-drop measurement via a clean-data linear probe
- repeated trials for stability
- threshold calibration from clean reference windows
- dataset abstraction so the workflow is not hardcoded to one split

The code supports:
- `cifar10`
- `cifar100`
- `imagenet_local` via an ImageFolder-style directory
- `fake` for offline structural smoke tests
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageEnhance
from scipy.stats import ks_2samp, pearsonr, spearmanr


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RAW_DATA_DIR = Path("data/raw_data")
DEFAULT_BLOODMNIST_ROOT = str(RAW_DATA_DIR)
DEFAULT_WATERBIRDS_ROOT = str(RAW_DATA_DIR / "waterbirds")
RESULTS_DIR = Path("outputs/results/controlled/default_run")
DEFAULT_MAX_FEATURE_DIMS = 256
MAX_FEATURE_DIMS = DEFAULT_MAX_FEATURE_DIMS


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


class AddGaussianNoise:
    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return clamp_tensor(x + self.std * torch.randn_like(x))


DRIFT_LIBRARY: Dict[str, Dict[str, Dict[str, float]]] = {
    "gaussian_noise": {
        "low": {"noise_std": 0.03},
        "medium": {"noise_std": 0.08},
        "high": {"noise_std": 0.15},
    },
    "motion_blur": {
        "low": {"kernel_size": 5},
        "medium": {"kernel_size": 9},
        "high": {"kernel_size": 13},
    },
    "brightness_shift": {
        "low": {"brightness_factor": 0.85},
        "medium": {"brightness_factor": 0.7},
        "high": {"brightness_factor": 0.5},
    },
}


@dataclass
class SampleBatch:
    images: List[Image.Image]
    labels: List[int]


@dataclass
class DatasetSpec:
    name: str
    train_dataset: object
    test_dataset: object
    num_classes: int
    note: str


@dataclass
class WaterbirdsSpec:
    name: str
    dataset: object
    metadata: object
    num_classes: int
    note: str


def base_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def ensure_rgb_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    image = Image.fromarray(np.asarray(image))
    return image.convert("RGB")


def apply_motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    tensor = T.ToTensor()(image).unsqueeze(0)
    kernel = torch.ones((3, 1, 1, kernel_size), dtype=tensor.dtype) / kernel_size
    blurred = F.conv2d(tensor, kernel, padding=(0, kernel_size // 2), groups=3)
    blurred = clamp_tensor(blurred.squeeze(0))
    return T.ToPILImage()(blurred)


def apply_pil_corruption(image: Image.Image, corruption_name: str, params: Dict[str, float]) -> Image.Image:
    if corruption_name == "motion_blur":
        return apply_motion_blur(image, int(params["kernel_size"]))
    if corruption_name == "brightness_shift":
        return ImageEnhance.Brightness(image).enhance(params["brightness_factor"])
    return image


def build_transform(corruption_name: str | None = None, severity: str | None = None) -> T.Compose:
    if corruption_name is None:
        return base_transform()

    params = DRIFT_LIBRARY[corruption_name][severity]
    steps: List[object] = [T.Resize((224, 224))]

    if corruption_name in {"motion_blur", "brightness_shift"}:
        steps.append(T.Lambda(lambda img: apply_pil_corruption(img, corruption_name, params)))

    steps.append(T.ToTensor())

    if corruption_name == "gaussian_noise":
        steps.append(AddGaussianNoise(params["noise_std"]))

    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(steps)


def dataset_to_batch(dataset, indices: Sequence[int]) -> SampleBatch:
    images = []
    labels = []
    for idx in indices:
        image, label = dataset[int(idx)]
        images.append(image)
        labels.append(int(label))
    return SampleBatch(images=images, labels=labels)


def sample_indices(length: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(length, size=min(sample_size, length), replace=False)


def load_cifar_dataset(name: str, train: bool, allow_download: bool):
    dataset_cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    try:
        return dataset_cls(root=str(RAW_DATA_DIR), train=train, download=False)
    except RuntimeError:
        if allow_download:
            return dataset_cls(root=str(RAW_DATA_DIR), train=train, download=True)
        split = "train" if train else "test"
        raise RuntimeError(
            f"{name} {split} split is not available locally in {RAW_DATA_DIR} and downloads are disabled. "
            "Run once with --download in a network-enabled environment."
        )


class MedMNISTRGBDataset:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        image = ensure_rgb_pil(image)
        if isinstance(label, np.ndarray):
            label = int(np.asarray(label).squeeze())
        elif torch.is_tensor(label):
            label = int(label.squeeze().item())
        else:
            label = int(label)
        return image, label


def load_bloodmnist_dataset(split: str, root: str, allow_download: bool):
    try:
        from medmnist import BloodMNIST
    except ImportError as exc:
        raise RuntimeError(
            "BloodMNIST support requires the `medmnist` package. Install it with `pip install medmnist`."
        ) from exc

    try:
        dataset = BloodMNIST(split=split, root=root, as_rgb=True, download=allow_download)
    except Exception as exc:
        raise RuntimeError(
            "BloodMNIST could not be loaded. If the dataset is not cached locally, rerun with --download."
        ) from exc
    return MedMNISTRGBDataset(dataset)


def build_fake_dataset(num_samples: int, num_classes: int):
    to_pil = T.ToPILImage()
    dataset = []
    for _ in range(num_samples):
        image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        label = int(torch.randint(0, num_classes, (1,)).item())
        dataset.append((to_pil(image), label))
    return dataset


class WaterbirdsDataset:
    def __init__(self, root: Path, metadata, image_col: str) -> None:
        self.root = root
        self.metadata = metadata.reset_index(drop=True)
        self.image_col = image_col

    def __len__(self) -> int:
        return len(self.metadata)

    def resolve_image_path(self, relative_path: str) -> Path:
        direct = self.root / relative_path
        if direct.exists():
            return direct
        images_dir = self.root / "images" / relative_path
        if images_dir.exists():
            return images_dir
        raise FileNotFoundError(
            f"Could not resolve Waterbirds image path `{relative_path}` under {self.root}."
        )

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[int(idx)]
        image_path = self.resolve_image_path(str(row[self.image_col]))
        image = Image.open(image_path).convert("RGB")
        label = int(row["y"])
        return image, label


def load_waterbirds_spec(root: str) -> WaterbirdsSpec:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Waterbirds support requires pandas.") from exc

    root_path = Path(root)
    metadata_path = root_path / "metadata.csv"
    if not metadata_path.exists():
        raise RuntimeError("Waterbirds requires a metadata.csv file in the dataset root.")

    metadata = pd.read_csv(metadata_path)
    image_col = next((col for col in ["img_filename", "filepath", "path"] if col in metadata.columns), None)
    if image_col is None:
        raise RuntimeError("Waterbirds metadata must include one of: img_filename, filepath, path.")

    if "y" not in metadata.columns:
        raise RuntimeError("Waterbirds metadata must include a `y` label column.")
    if "place" not in metadata.columns:
        raise RuntimeError("Waterbirds metadata must include a `place` background column.")
    if "split" not in metadata.columns:
        raise RuntimeError("Waterbirds metadata must include a `split` column.")

    split_map = {0: "train", 1: "val", 2: "test", "train": "train", "val": "val", "test": "test"}
    metadata["split_name"] = metadata["split"].map(lambda value: split_map.get(value, str(value)))
    metadata["aligned"] = (metadata["y"].astype(int) == metadata["place"].astype(int)).astype(int)

    label_names = {0: "landbird", 1: "waterbird"}
    place_names = {0: "land", 1: "water"}
    metadata["label_name"] = metadata["y"].map(lambda value: label_names.get(int(value), f"label_{value}"))
    metadata["place_name"] = metadata["place"].map(lambda value: place_names.get(int(value), f"place_{value}"))
    metadata["group_name"] = metadata["label_name"] + "_" + metadata["place_name"]
    metadata["alignment_name"] = metadata["aligned"].map({1: "matched", 0: "mismatched"})

    dataset = WaterbirdsDataset(root_path, metadata, image_col=image_col)
    return WaterbirdsSpec(
        name="waterbirds",
        dataset=dataset,
        metadata=metadata,
        num_classes=2,
        note=(
            "Waterbirds natural-shift benchmark using metadata-defined environments. "
            "Train features are fit on train split labels; reference drift calibration uses matched validation groups."
        ),
    )


def load_dataset_spec(
    dataset_name: str,
    allow_download: bool = False,
    imagenet_root: str | None = None,
    bloodmnist_root: str = DEFAULT_BLOODMNIST_ROOT,
    fake_train_size: int = 128,
    fake_test_size: int = 96,
) -> DatasetSpec:
    if dataset_name == "cifar10":
        return DatasetSpec(
            name="cifar10",
            train_dataset=load_cifar_dataset("cifar10", train=True, allow_download=allow_download),
            test_dataset=load_cifar_dataset("cifar10", train=False, allow_download=allow_download),
            num_classes=10,
            note="CIFAR-10 benchmark split",
        )

    if dataset_name == "cifar100":
        return DatasetSpec(
            name="cifar100",
            train_dataset=load_cifar_dataset("cifar100", train=True, allow_download=allow_download),
            test_dataset=load_cifar_dataset("cifar100", train=False, allow_download=allow_download),
            num_classes=100,
            note="CIFAR-100 benchmark split",
        )

    if dataset_name == "bloodmnist":
        return DatasetSpec(
            name="bloodmnist",
            train_dataset=load_bloodmnist_dataset("train", root=bloodmnist_root, allow_download=allow_download),
            test_dataset=load_bloodmnist_dataset("test", root=bloodmnist_root, allow_download=allow_download),
            num_classes=8,
            note="BloodMNIST medical benchmark split with synthetic corruption evaluation.",
        )

    if dataset_name == "imagenet_local":
        if not imagenet_root:
            raise RuntimeError("imagenet_local requires --imagenet-root pointing to an ImageFolder dataset.")
        root = Path(imagenet_root)
        train_root = root / "train"
        val_root = root / "val"
        if not train_root.exists() or not val_root.exists():
            raise RuntimeError(
                "imagenet_local expects an ImageFolder layout with 'train/' and 'val/' subdirectories."
            )
        train_dataset = torchvision.datasets.ImageFolder(train_root)
        test_dataset = torchvision.datasets.ImageFolder(val_root)
        return DatasetSpec(
            name="imagenet_local",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_classes=len(train_dataset.classes),
            note=f"Local ImageNet-style dataset from {root}",
        )

    if dataset_name == "fake":
        return DatasetSpec(
            name="fake",
            train_dataset=build_fake_dataset(fake_train_size, num_classes=10),
            test_dataset=build_fake_dataset(fake_test_size, num_classes=10),
            num_classes=10,
            note="Synthetic smoke-test dataset",
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_names: Sequence[str]) -> None:
        self.activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
        self.handles = []
        for name, module in model.named_modules():
            if name in layer_names:
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def inner(_module, _inputs, output):
            if output.ndim == 4:
                output = output.mean(dim=(2, 3))
            self.activations[name].append(output.detach().cpu())

        return inner

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def batched_images(images: Sequence[Image.Image], transform: T.Compose, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        yield torch.stack([transform(image) for image in batch])


def extract_features(
    model: nn.Module,
    images: Sequence[Image.Image],
    transform: T.Compose,
    layer_names: Sequence[str],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    extractor = FeatureExtractor(model, layer_names)
    model.eval()
    with torch.no_grad():
        for batch in batched_images(images, transform, batch_size):
            model(batch.to(DEVICE))
    outputs = {name: torch.cat(chunks).numpy() for name, chunks in extractor.activations.items()}
    extractor.close()
    return outputs


def subsample_feature_dims(
    x: np.ndarray,
    y: np.ndarray,
    max_feature_dims: int = DEFAULT_MAX_FEATURE_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if max_feature_dims and x.shape[1] > max_feature_dims:
        dim_idx = np.linspace(0, x.shape[1] - 1, max_feature_dims, dtype=int)
        x = x[:, dim_idx]
        y = y[:, dim_idx]

    return x, y


def compute_ks_statistic(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return compute_ks_statistic_fast(x, y, max_feature_dims=MAX_FEATURE_DIMS)


def compute_ks_statistic_fast(
    x: np.ndarray,
    y: np.ndarray,
    max_feature_dims: int = MAX_FEATURE_DIMS,
) -> Dict[str, float]:
    x, y = subsample_feature_dims(x, y, max_feature_dims=max_feature_dims)

    stats = []
    pvalues = []
    for dim in range(x.shape[1]):
        stat, pvalue = ks_2samp(x[:, dim], y[:, dim])
        stats.append(stat)
        pvalues.append(pvalue)

    return {
        "ks_mean": float(np.mean(stats)),
        "ks_max": float(np.max(stats)),
        "pvalue_mean": float(np.mean(pvalues)),
    }


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1, keepdim=True).T
    return torch.clamp(x_norm + y_norm - 2 * x @ y.T, min=0.0)


def compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 256,
    max_feature_dims: int = MAX_FEATURE_DIMS,
    estimator: str = "biased",
) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x, y = subsample_feature_dims(x, y, max_feature_dims=max_feature_dims)

    if x.shape[0] > max_samples:
        x = x[np.linspace(0, x.shape[0] - 1, max_samples, dtype=int)]
    if y.shape[0] > max_samples:
        y = y[np.linspace(0, y.shape[0] - 1, max_samples, dtype=int)]

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    pooled = torch.cat([x_t, y_t], dim=0)
    sq_dists = _pairwise_sq_dists(pooled, pooled)
    sigma2 = torch.median(sq_dists[sq_dists > 0])
    if torch.isnan(sigma2) or sigma2 <= 0:
        sigma2 = torch.tensor(1.0)

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-_pairwise_sq_dists(a, b) / (2 * sigma2))

    k_xx = kernel(x_t, x_t)
    k_yy = kernel(y_t, y_t)
    k_xy = kernel(x_t, y_t)

    n = x_t.shape[0]
    m = y_t.shape[0]
    if n < 2 or m < 2:
        return 0.0

    if estimator == "unbiased":
        mmd2 = (
            (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
            + (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
            - 2 * k_xy.mean()
        )
    elif estimator == "biased":
        # The biased estimator stays non-negative more reliably for
        # clean-vs-clean calibration windows, which makes thresholding
        # substantially more stable for monitoring use.
        mmd2 = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    else:
        raise ValueError(f"Unsupported MMD estimator: {estimator}")

    return float(torch.sqrt(torch.clamp(mmd2, min=0.0)))


class LinearProbe(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: Sequence[int],
    num_classes: int,
    epochs: int = 12,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
) -> LinearProbe:
    x = torch.from_numpy(train_features).float()
    y = torch.tensor(train_labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    probe = LinearProbe(train_features.shape[1], num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    probe.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return probe.eval()


def evaluate_accuracy(probe: LinearProbe, features: np.ndarray, labels: Sequence[int], batch_size: int = 128) -> float:
    x = torch.from_numpy(features).float()
    y = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = probe(batch_x.to(DEVICE))
            preds = torch.argmax(logits, dim=1).cpu()
            correct += int((preds == batch_y).sum())
            total += int(batch_y.shape[0])

    return correct / total if total else 0.0


def calibrate_clean_thresholds(
    clean_reference: np.ndarray,
    window_size: int,
    num_windows: int,
    seed: int,
    quantile: float = 0.95,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    count = clean_reference.shape[0]
    if count < max(window_size * 2, 4):
        return {"ks_threshold": math.nan, "mmd_threshold": math.nan}

    ks_values = []
    mmd_values = []
    for _ in range(num_windows):
        first = rng.choice(count, size=window_size, replace=False)
        second = rng.choice(count, size=window_size, replace=False)
        x = clean_reference[first]
        y = clean_reference[second]
        ks_values.append(compute_ks_statistic_fast(x, y, max_feature_dims=MAX_FEATURE_DIMS)["ks_mean"])
        mmd_values.append(compute_mmd(x, y, estimator="biased", max_feature_dims=MAX_FEATURE_DIMS))

    ks_threshold = float(np.quantile(ks_values, quantile))
    mmd_threshold = float(np.quantile(mmd_values, quantile))

    # Small clean windows can occasionally collapse to numerically tiny MMD
    # values. When that happens, use a dispersion-based fallback rather than
    # keeping a zero threshold that would trigger meaningless alerts.
    if mmd_threshold <= 1e-8:
        mmd_threshold = float(max(np.mean(mmd_values) + 2 * np.std(mmd_values), np.max(mmd_values), 1e-6))

    return {
        "ks_threshold": ks_threshold,
        "mmd_threshold": mmd_threshold,
    }


def safe_corr(x: np.ndarray, y: np.ndarray, fn) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(fn(x, y).statistic)


def correlation_summary(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    accuracy_drop = np.array([row["accuracy_drop"] for row in rows], dtype=float)
    ks_mean = np.array([row["ks_mean"] for row in rows], dtype=float)
    mmd = np.array([row["mmd"] for row in rows], dtype=float)

    return {
        "pearson_accuracy_vs_ks": safe_corr(accuracy_drop, ks_mean, pearsonr),
        "pearson_accuracy_vs_mmd": safe_corr(accuracy_drop, mmd, pearsonr),
        "spearman_accuracy_vs_ks": safe_corr(accuracy_drop, ks_mean, spearmanr),
        "spearman_accuracy_vs_mmd": safe_corr(accuracy_drop, mmd, spearmanr),
    }


def aggregate_rows(rows: Sequence[Dict[str, float]], group_keys: Sequence[str]) -> List[Dict[str, float]]:
    grouped: Dict[tuple, List[Dict[str, float]]] = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    numeric_keys = [key for key in rows[0].keys() if key not in group_keys]
    for row in rows[1:]:
        for key in row.keys():
            if key not in group_keys and key not in numeric_keys:
                numeric_keys.append(key)
    for key, group in grouped.items():
        out = {name: value for name, value in zip(group_keys, key)}
        for metric in numeric_keys:
            values = []
            for item in group:
                value = item.get(metric)
                if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
                    values.append(float(value))
            if values:
                out[metric] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values))
        summary_rows.append(out)

    return summary_rows


def load_models(random_weights: bool = False) -> Dict[str, Dict[str, object]]:
    try:
        resnet_weights = None if random_weights else models.ResNet50_Weights.IMAGENET1K_V1
        efficientnet_weights = None if random_weights else models.EfficientNet_B0_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=resnet_weights).to(DEVICE).eval()
        efficientnet = models.efficientnet_b0(weights=efficientnet_weights).to(DEVICE).eval()
    except Exception as exc:
        if random_weights:
            raise
        raise RuntimeError(
            "Pretrained model weights are not available locally and could not be downloaded. "
            "Run once in a network-enabled environment or use --random-weights for a structural smoke test."
        ) from exc

    return {
        "ResNet-50": {
            "model": resnet,
            "layers": {
                "layer1": "Layer 1 (Early)",
                "layer2": "Layer 2 (Shallow)",
                "layer3": "Layer 3 (Deep)",
                "layer4": "Layer 4 (Final)",
            },
            "probe_layer": "layer4",
        },
        "EfficientNet-B0": {
            "model": efficientnet,
            "layers": {
                "features.3": "MBConv Block 1",
                "features.4": "MBConv Block 2",
                "features.6": "MBConv Block 4",
                "features.8": "MBConv Block 6 (Final)",
            },
            "probe_layer": "features.8",
        },
    }


def summarize_findings(summary_rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    grouped = {}
    for row in summary_rows:
        key = (row["dataset"], row["model"])
        grouped.setdefault(key, []).append(row)

    findings = []
    for (dataset_name, model_name), rows in grouped.items():
        best_drop = max(rows, key=lambda row: row.get("accuracy_drop", float("-inf")))
        best_mmd = max(rows, key=lambda row: row.get("mmd", float("-inf")))
        findings.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "largest_accuracy_drop": {
                    "drift_type": best_drop["drift_type"],
                    "severity": best_drop["severity"],
                    "accuracy_drop": best_drop["accuracy_drop"],
                },
                "largest_mmd": {
                    "drift_type": best_mmd["drift_type"],
                    "severity": best_mmd["severity"],
                    "mmd": best_mmd["mmd"],
                },
            }
        )
    return {"high_level_findings": findings}


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_trial_batches(
    dataset_spec: DatasetSpec,
    train_size: int,
    test_size: int,
    trial_seed: int,
) -> tuple[SampleBatch, SampleBatch]:
    train_indices = sample_indices(len(dataset_spec.train_dataset), train_size, trial_seed)
    test_indices = sample_indices(len(dataset_spec.test_dataset), test_size, trial_seed + 10_000)
    return (
        dataset_to_batch(dataset_spec.train_dataset, train_indices),
        dataset_to_batch(dataset_spec.test_dataset, test_indices),
    )


def sample_fixed_indices(indices: Sequence[int], max_samples: int, seed: int) -> List[int]:
    indices = list(indices)
    if len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=max_samples, replace=False)
    return [int(idx) for idx in selected]


def run_controlled_experiments(
    datasets: Sequence[str],
    train_size: int,
    test_size: int,
    batch_size: int,
    probe_epochs: int,
    trials: int,
    output_dir: Path,
    allow_download: bool = False,
    imagenet_root: str | None = None,
    bloodmnist_root: str = DEFAULT_BLOODMNIST_ROOT,
    random_weights: bool = False,
) -> Dict[str, object]:
    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = load_models(random_weights=random_weights)
    raw_rows: List[Dict[str, object]] = []
    correlations = []
    calibration_rows = []
    dataset_notes = {}

    for dataset_idx, dataset_name in enumerate(datasets):
        dataset_spec = load_dataset_spec(
            dataset_name,
            allow_download=allow_download,
            imagenet_root=imagenet_root,
            bloodmnist_root=bloodmnist_root,
            fake_train_size=max(train_size, 64),
            fake_test_size=max(test_size, 48),
        )
        dataset_notes[dataset_name] = dataset_spec.note

        for trial_idx in range(trials):
            trial_seed = SEED + dataset_idx * 1000 + trial_idx * 101
            train_batch, test_batch = build_trial_batches(dataset_spec, train_size, test_size, trial_seed)

            for model_name, cfg in models_cfg.items():
                model = cfg["model"]
                layer_names = list(cfg["layers"].keys())
                probe_layer = cfg["probe_layer"]

                train_features = extract_features(model, train_batch.images, base_transform(), [probe_layer], batch_size)
                probe = train_linear_probe(
                    train_features[probe_layer],
                    train_batch.labels,
                    num_classes=dataset_spec.num_classes,
                    epochs=probe_epochs,
                    batch_size=batch_size,
                )

                clean_features = extract_features(model, test_batch.images, base_transform(), layer_names, batch_size)
                clean_probe_features = clean_features[probe_layer]
                clean_accuracy = evaluate_accuracy(probe, clean_probe_features, test_batch.labels)
                thresholds = calibrate_clean_thresholds(
                    clean_probe_features,
                    window_size=min(32, max(8, clean_probe_features.shape[0] // 2)),
                    num_windows=12,
                    seed=trial_seed,
                )
                calibration_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "trial": trial_idx,
                        "ks_threshold": thresholds["ks_threshold"],
                        "mmd_threshold": thresholds["mmd_threshold"],
                        "clean_accuracy": clean_accuracy,
                    }
                )

                for drift_type in DRIFT_LIBRARY.keys():
                    for severity in ("low", "medium", "high"):
                        drift_transform = build_transform(drift_type, severity)
                        drift_features = extract_features(model, test_batch.images, drift_transform, layer_names, batch_size)
                        drift_probe_features = drift_features[probe_layer]
                        drift_accuracy = evaluate_accuracy(probe, drift_probe_features, test_batch.labels)

                        row: Dict[str, object] = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "trial": trial_idx,
                            "drift_type": drift_type,
                            "severity": severity,
                            "clean_accuracy": float(clean_accuracy),
                            "drift_accuracy": float(drift_accuracy),
                            "accuracy_drop": float(clean_accuracy - drift_accuracy),
                            "probe_layer": probe_layer,
                            "ks_threshold": thresholds["ks_threshold"],
                            "mmd_threshold": thresholds["mmd_threshold"],
                        }

                        for layer_name in layer_names:
                            ks_stats = compute_ks_statistic_fast(
                                clean_features[layer_name],
                                drift_features[layer_name],
                                max_feature_dims=MAX_FEATURE_DIMS,
                            )
                            mmd_value = compute_mmd(
                                clean_features[layer_name],
                                drift_features[layer_name],
                                max_feature_dims=MAX_FEATURE_DIMS,
                            )
                            row[f"{layer_name}_ks_mean"] = ks_stats["ks_mean"]
                            row[f"{layer_name}_ks_max"] = ks_stats["ks_max"]
                            row[f"{layer_name}_mmd"] = mmd_value

                            if layer_name == probe_layer:
                                row["ks_mean"] = ks_stats["ks_mean"]
                                row["ks_max"] = ks_stats["ks_max"]
                                row["pvalue_mean"] = ks_stats["pvalue_mean"]
                                row["mmd"] = mmd_value
                                row["ks_alert"] = bool(
                                    not math.isnan(thresholds["ks_threshold"]) and ks_stats["ks_mean"] > thresholds["ks_threshold"]
                                )
                                row["mmd_alert"] = bool(
                                    not math.isnan(thresholds["mmd_threshold"]) and mmd_value > thresholds["mmd_threshold"]
                                )

                        raw_rows.append(row)

        for model_name in models_cfg.keys():
            model_rows = [row for row in raw_rows if row["dataset"] == dataset_name and row["model"] == model_name]
            correlations.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    **correlation_summary(model_rows),
                }
            )

    summary_rows = aggregate_rows(raw_rows, ["dataset", "model", "drift_type", "severity", "probe_layer"])
    findings = summarize_findings(summary_rows)

    raw_csv = output_dir / "experiment_results_raw.csv"
    summary_csv = output_dir / "experiment_results_summary.csv"
    corr_csv = output_dir / "correlation_summary.csv"
    calib_csv = output_dir / "calibration_summary.csv"
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(corr_csv, correlations)
    write_csv(calib_csv, calibration_rows)

    summary = {
        "seed": SEED,
        "datasets": list(datasets),
        "dataset_notes": dataset_notes,
        "train_size": train_size,
        "test_size": test_size,
        "trials": trials,
        "random_weights": random_weights,
        "output_files": {
            "raw_results": str(raw_csv.resolve()),
            "summary_results": str(summary_csv.resolve()),
            "correlations": str(corr_csv.resolve()),
            "calibration": str(calib_csv.resolve()),
        },
        "correlations": correlations,
        **findings,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize_waterbirds_findings(summary_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["model"], []).append(row)

    findings = []
    for model_name, rows in grouped.items():
        best_drop = max(rows, key=lambda row: row.get("accuracy_drop", float("-inf")))
        best_mmd = max(rows, key=lambda row: row.get("mmd", float("-inf")))
        findings.append(
            {
                "dataset": "waterbirds",
                "model": model_name,
                "largest_accuracy_drop": {
                    "target_group": best_drop["target_group"],
                    "accuracy_drop": best_drop["accuracy_drop"],
                },
                "largest_mmd": {
                    "target_group": best_mmd["target_group"],
                    "mmd": best_mmd["mmd"],
                },
            }
        )
    return {"high_level_findings": findings}


def run_waterbirds_experiment(
    waterbirds_root: str,
    train_size: int,
    test_size: int,
    batch_size: int,
    probe_epochs: int,
    trials: int,
    output_dir: Path,
    random_weights: bool = False,
) -> Dict[str, object]:
    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_waterbirds_spec(waterbirds_root)
    models_cfg = load_models(random_weights=random_weights)
    metadata = spec.metadata
    dataset = spec.dataset

    train_indices_all = metadata.index[metadata["split_name"] == "train"].tolist()
    reference_indices_all = metadata.index[
        (metadata["split_name"] == "val") & (metadata["aligned"] == 1)
    ].tolist()
    if not train_indices_all or not reference_indices_all:
        raise RuntimeError(
            "Waterbirds requires non-empty train split and matched validation reference subset."
        )

    target_specs = [
        ("val_matched", (metadata["split_name"] == "val") & (metadata["aligned"] == 1)),
        ("val_mismatched", (metadata["split_name"] == "val") & (metadata["aligned"] == 0)),
        ("test_matched", (metadata["split_name"] == "test") & (metadata["aligned"] == 1)),
        ("test_mismatched", (metadata["split_name"] == "test") & (metadata["aligned"] == 0)),
    ]
    for split_name in ["val", "test"]:
        for group_name in sorted(metadata["group_name"].unique()):
            target_specs.append(
                (
                    f"{split_name}_{group_name}",
                    (metadata["split_name"] == split_name) & (metadata["group_name"] == group_name),
                )
            )

    raw_rows: List[Dict[str, object]] = []
    correlations = []
    calibration_rows = []

    for trial_idx in range(trials):
        trial_seed = SEED + trial_idx * 101
        train_indices = sample_fixed_indices(train_indices_all, train_size, trial_seed)
        reference_indices = sample_fixed_indices(reference_indices_all, test_size, trial_seed + 1)
        train_batch = dataset_to_batch(dataset, train_indices)
        reference_batch = dataset_to_batch(dataset, reference_indices)

        for model_name, cfg in models_cfg.items():
            model = cfg["model"]
            layer_names = list(cfg["layers"].keys())
            probe_layer = cfg["probe_layer"]

            train_features = extract_features(model, train_batch.images, base_transform(), [probe_layer], batch_size)
            probe = train_linear_probe(
                train_features[probe_layer],
                train_batch.labels,
                num_classes=spec.num_classes,
                epochs=probe_epochs,
                batch_size=batch_size,
            )

            reference_features = extract_features(model, reference_batch.images, base_transform(), layer_names, batch_size)
            reference_probe_features = reference_features[probe_layer]
            reference_accuracy = evaluate_accuracy(probe, reference_probe_features, reference_batch.labels)
            thresholds = calibrate_clean_thresholds(
                reference_probe_features,
                window_size=min(32, max(8, reference_probe_features.shape[0] // 2)),
                num_windows=12,
                seed=trial_seed,
            )
            calibration_rows.append(
                {
                    "dataset": "waterbirds",
                    "model": model_name,
                    "trial": trial_idx,
                    "reference_group": "val_matched",
                    "ks_threshold": thresholds["ks_threshold"],
                    "mmd_threshold": thresholds["mmd_threshold"],
                    "reference_accuracy": reference_accuracy,
                }
            )

            for target_group, mask in target_specs:
                indices_all = metadata.index[mask].tolist()
                if not indices_all:
                    continue
                target_indices = sample_fixed_indices(indices_all, test_size, trial_seed + len(target_group))
                target_batch = dataset_to_batch(dataset, target_indices)
                target_features = extract_features(model, target_batch.images, base_transform(), layer_names, batch_size)
                target_probe_features = target_features[probe_layer]
                target_accuracy = evaluate_accuracy(probe, target_probe_features, target_batch.labels)

                split_name = metadata.loc[target_indices[0], "split_name"]
                alignment_name = (
                    metadata.loc[target_indices, "alignment_name"].mode().iloc[0]
                    if "alignment_name" in metadata.columns
                    else "mixed"
                )

                row: Dict[str, object] = {
                    "dataset": "waterbirds",
                    "model": model_name,
                    "trial": trial_idx,
                    "reference_group": "val_matched",
                    "target_group": target_group,
                    "target_split": split_name,
                    "alignment": alignment_name,
                    "clean_accuracy": float(reference_accuracy),
                    "drift_accuracy": float(target_accuracy),
                    "accuracy_drop": float(reference_accuracy - target_accuracy),
                    "probe_layer": probe_layer,
                    "ks_threshold": thresholds["ks_threshold"],
                    "mmd_threshold": thresholds["mmd_threshold"],
                    "group_size": len(target_indices),
                }

                for layer_name in layer_names:
                    ks_stats = compute_ks_statistic_fast(
                        reference_features[layer_name],
                        target_features[layer_name],
                        max_feature_dims=MAX_FEATURE_DIMS,
                    )
                    mmd_value = compute_mmd(
                        reference_features[layer_name],
                        target_features[layer_name],
                        max_feature_dims=MAX_FEATURE_DIMS,
                    )
                    row[f"{layer_name}_ks_mean"] = ks_stats["ks_mean"]
                    row[f"{layer_name}_ks_max"] = ks_stats["ks_max"]
                    row[f"{layer_name}_mmd"] = mmd_value

                    if layer_name == probe_layer:
                        row["ks_mean"] = ks_stats["ks_mean"]
                        row["ks_max"] = ks_stats["ks_max"]
                        row["pvalue_mean"] = ks_stats["pvalue_mean"]
                        row["mmd"] = mmd_value
                        row["ks_alert"] = bool(
                            not math.isnan(thresholds["ks_threshold"]) and ks_stats["ks_mean"] > thresholds["ks_threshold"]
                        )
                        row["mmd_alert"] = bool(
                            not math.isnan(thresholds["mmd_threshold"]) and mmd_value > thresholds["mmd_threshold"]
                        )

                raw_rows.append(row)

    for model_name in models_cfg.keys():
        model_rows = [row for row in raw_rows if row["model"] == model_name]
        correlations.append(
            {
                "dataset": "waterbirds",
                "model": model_name,
                **correlation_summary(model_rows),
            }
        )

    summary_rows = aggregate_rows(
        raw_rows,
        ["dataset", "model", "reference_group", "target_group", "target_split", "alignment", "probe_layer"],
    )
    findings = summarize_waterbirds_findings(summary_rows)

    raw_csv = output_dir / "experiment_results_raw.csv"
    summary_csv = output_dir / "experiment_results_summary.csv"
    corr_csv = output_dir / "correlation_summary.csv"
    calib_csv = output_dir / "calibration_summary.csv"
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(corr_csv, correlations)
    write_csv(calib_csv, calibration_rows)

    summary = {
        "seed": SEED,
        "datasets": ["waterbirds"],
        "dataset_notes": {"waterbirds": spec.note},
        "train_size": train_size,
        "test_size": test_size,
        "trials": trials,
        "random_weights": random_weights,
        "waterbirds_design": {
            "reference_group": "matched validation subset",
            "targets": "validation/test alignment groups and class-background environments",
            "claim_scope": "natural distribution shift and spurious-correlation sensitivity, not synthetic corruption robustness",
        },
        "output_files": {
            "raw_results": str(raw_csv.resolve()),
            "summary_results": str(summary_csv.resolve()),
            "correlations": str(corr_csv.resolve()),
            "calibration": str(calib_csv.resolve()),
        },
        "correlations": correlations,
        **findings,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_experiments(
    datasets: Sequence[str],
    train_size: int,
    test_size: int,
    batch_size: int,
    probe_epochs: int,
    trials: int,
    output_dir: Path,
    allow_download: bool = False,
    imagenet_root: str | None = None,
    bloodmnist_root: str = DEFAULT_BLOODMNIST_ROOT,
    waterbirds_root: str | None = None,
    random_weights: bool = False,
) -> Dict[str, object]:
    if "waterbirds" in datasets:
        if len(datasets) != 1:
            raise RuntimeError(
                "Run Waterbirds separately from controlled benchmarks because it uses a different evaluation schema."
            )
        if not waterbirds_root:
            raise RuntimeError("Waterbirds evaluation requires --waterbirds-root pointing to the dataset root.")
        return run_waterbirds_experiment(
            waterbirds_root=waterbirds_root,
            train_size=train_size,
            test_size=test_size,
            batch_size=batch_size,
            probe_epochs=probe_epochs,
            trials=trials,
            output_dir=output_dir,
            random_weights=random_weights,
        )

    return run_controlled_experiments(
        datasets=datasets,
        train_size=train_size,
        test_size=test_size,
        batch_size=batch_size,
        probe_epochs=probe_epochs,
        trials=trials,
        output_dir=output_dir,
        allow_download=allow_download,
        imagenet_root=imagenet_root,
        bloodmnist_root=bloodmnist_root,
        random_weights=random_weights,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift-vs-accuracy experiments for the capstone project.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10"],
        choices=["cifar10", "cifar100", "bloodmnist", "waterbirds", "imagenet_local", "fake"],
        help="Datasets to evaluate.",
    )
    parser.add_argument("--imagenet-root", type=str, default=None, help="Local ImageFolder root for imagenet_local.")
    parser.add_argument(
        "--bloodmnist-root",
        type=str,
        default=DEFAULT_BLOODMNIST_ROOT,
        help="Root directory for BloodMNIST cache/download.",
    )
    parser.add_argument(
        "--waterbirds-root",
        type=str,
        default=DEFAULT_WATERBIRDS_ROOT,
        help="Root directory for Waterbirds dataset with metadata.csv.",
    )
    parser.add_argument("--train-size", type=int, default=500, help="Number of clean training samples per trial.")
    parser.add_argument("--test-size", type=int, default=300, help="Number of clean/drift test samples per trial.")
    parser.add_argument("--batch-size", type=int, default=32, help="Feature extraction batch size.")
    parser.add_argument("--probe-epochs", type=int, default=12, help="Linear probe training epochs.")
    parser.add_argument("--trials", type=int, default=3, help="Repeated trials per dataset/model pair.")
    parser.add_argument(
        "--max-feature-dims",
        type=int,
        default=DEFAULT_MAX_FEATURE_DIMS,
        help="Maximum feature dimensions used for K-S and MMD statistics per layer.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for CSV/JSON results.")
    parser.add_argument("--download", action="store_true", help="Allow dataset download when supported.")
    parser.add_argument("--random-weights", action="store_true", help="Use randomly initialized backbones for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MAX_FEATURE_DIMS
    MAX_FEATURE_DIMS = args.max_feature_dims
    summary = run_experiments(
        datasets=args.datasets,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        probe_epochs=args.probe_epochs,
        trials=args.trials,
        output_dir=args.output_dir,
        allow_download=args.download,
        imagenet_root=args.imagenet_root,
        bloodmnist_root=args.bloodmnist_root,
        waterbirds_root=args.waterbirds_root,
        random_weights=args.random_weights,
    )

    print("\n=== Correlation Summary ===")
    for row in summary["correlations"]:
        print(f"{row['dataset']} / {row['model']}")
        for key, value in row.items():
            if key in {"dataset", "model"}:
                continue
            print(f"  {key}: {value:.4f}" if not math.isnan(value) else f"  {key}: nan")

    print("\n=== High-Level Findings ===")
    for item in summary["high_level_findings"]:
        print(f"{item['dataset']} / {item['model']}")
        largest_drop = item["largest_accuracy_drop"]
        largest_mmd = item["largest_mmd"]
        if "drift_type" in largest_drop:
            print(
                f"  Largest accuracy drop: {largest_drop['drift_type']} / {largest_drop['severity']} "
                f"({largest_drop['accuracy_drop']:.4f})"
            )
            print(
                f"  Largest MMD: {largest_mmd['drift_type']} / {largest_mmd['severity']} "
                f"({largest_mmd['mmd']:.4f})"
            )
        else:
            print(
                f"  Largest accuracy drop: {largest_drop['target_group']} "
                f"({largest_drop['accuracy_drop']:.4f})"
            )
            print(
                f"  Largest MMD: {largest_mmd['target_group']} "
                f"({largest_mmd['mmd']:.4f})"
            )

    print("\nSaved outputs:")
    for name, path in summary["output_files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
