"""Shared plotting style and path helpers for CAP visualizations."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE_DIR = PROJECT_ROOT / ".mpl-cache"
XDG_CACHE_DIR = PROJECT_ROOT / ".cache"

SEVERITY_ORDER = ["low", "medium", "high"]
DATASET_COLORS = {
    "cifar10": "#d1495b",
    "bloodmnist": "#2e86ab",
    "waterbirds": "#6a994e",
}
MODEL_COLORS = {
    "ResNet-50": "#1b4332",
    "EfficientNet-B0": "#9c6644",
}
DRIFT_COLORS = {
    "gaussian_noise": "#d1495b",
    "motion_blur": "#2e86ab",
    "brightness_shift": "#edae49",
}
ALIGNMENT_COLORS = {
    "matched": "#2a9d8f",
    "mismatched": "#e76f51",
}


def configure_plot_environment() -> None:
    """Point matplotlib caches inside the repository for reproducible runs."""
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR.resolve()))


def ensure_dir(path: Path) -> None:
    """Create a directory tree when it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def apply_publication_style(plt_module) -> None:
    """Apply a consistent, publication-friendly style across all figures."""
    plt_module.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "figure.titlesize": 16,
            "legend.frameon": False,
            "font.size": 10,
            "grid.alpha": 0.24,
        }
    )
