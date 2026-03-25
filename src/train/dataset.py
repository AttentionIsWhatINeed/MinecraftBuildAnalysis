import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DatasetBundle:
    train_samples: List[Dict]
    val_samples: List[Dict]
    test_samples: List[Dict]
    tag_to_idx: Dict[str, int]
    idx_to_tag: List[str]


class MinecraftImageDataset(Dataset):
    def __init__(self, samples: List[Dict], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.samples[idx]
        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")
        return self.transform(image), row["target"]


def _load_json_builds(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "builds" in raw:
        return raw["builds"]
    if isinstance(raw, list):
        return raw

    raise ValueError(f"Unsupported dataset format: {path}")


def _resolve_image_path(path_str: str, root: Path) -> Path:
    # JSON uses Windows-style separators; normalize for local runtime.
    normalized = path_str.replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        return p
    return root / p


def _expand_to_image_samples(builds: List[Dict], tag_to_idx: Dict[str, int], root: Path) -> List[Dict]:
    samples: List[Dict] = []
    for build in builds:
        tags = [t for t in build.get("tags", []) if t in tag_to_idx]
        if not tags:
            continue

        multi_hot = torch.zeros(len(tag_to_idx), dtype=torch.float32)
        for t in tags:
            multi_hot[tag_to_idx[t]] = 1.0

        for img_path in build.get("local_image_paths", []):
            resolved = _resolve_image_path(img_path, root)
            if resolved.exists():
                samples.append({"image_path": resolved, "target": multi_hot.clone()})

    return samples


def build_datasets(train_json: Path, val_json: Path, test_json: Path, root: Path) -> DatasetBundle:
    train_builds = _load_json_builds(train_json)
    val_builds = _load_json_builds(val_json)
    test_builds = _load_json_builds(test_json)

    tag_set = set()
    for build in train_builds:
        for tag in build.get("tags", []):
            tag_set.add(tag)

    idx_to_tag = sorted(tag_set)
    if not idx_to_tag:
        raise ValueError("No tags found in training set.")

    tag_to_idx = {tag: idx for idx, tag in enumerate(idx_to_tag)}

    train_samples = _expand_to_image_samples(train_builds, tag_to_idx, root)
    val_samples = _expand_to_image_samples(val_builds, tag_to_idx, root)
    test_samples = _expand_to_image_samples(test_builds, tag_to_idx, root)

    if not train_samples:
        raise ValueError("No valid training image samples found.")

    return DatasetBundle(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag,
    )
