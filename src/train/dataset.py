import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DatasetBundle:
    train_build_samples: List[Dict]
    val_build_samples: List[Dict]
    test_build_samples: List[Dict]
    tag_to_idx: Dict[str, int]
    idx_to_tag: List[str]


class MinecraftBuildBagDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        transform: transforms.Compose,
        max_images_per_build: int,
        train_mode: bool,
        return_index: bool = False,
    ):
        if max_images_per_build <= 0:
            raise ValueError("max_images_per_build must be > 0")

        self.samples = samples
        self.transform = transform
        self.max_images_per_build = max_images_per_build
        self.train_mode = train_mode
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.samples)

    def _select_paths(self, paths: List[Path]) -> List[Path]:
        if len(paths) <= self.max_images_per_build:
            return paths

        if self.train_mode:
            chosen = torch.randperm(len(paths))[: self.max_images_per_build].tolist()
            return [paths[i] for i in chosen]

        return paths[: self.max_images_per_build]

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        selected_paths = self._select_paths(row["image_paths"])

        images: List[torch.Tensor] = []
        for image_path in selected_paths:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            images.append(self.transform(image))

        sample_shape = images[0].shape
        bag = torch.zeros((self.max_images_per_build, *sample_shape), dtype=images[0].dtype)
        mask = torch.zeros(self.max_images_per_build, dtype=torch.bool)

        for i, image in enumerate(images):
            bag[i] = image
            mask[i] = True

        target = row["target"].clone()

        if self.return_index:
            return bag, mask, target, idx
        return bag, mask, target


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


def _expand_to_build_samples(builds: List[Dict], tag_to_idx: Dict[str, int], root: Path) -> List[Dict]:
    samples: List[Dict] = []
    for build in builds:
        tags = [t for t in build.get("tags", []) if t in tag_to_idx]
        if not tags:
            continue

        multi_hot = torch.zeros(len(tag_to_idx), dtype=torch.float32)
        for t in tags:
            multi_hot[tag_to_idx[t]] = 1.0

        image_paths: List[Path] = []
        for img_path in build.get("local_image_paths", []):
            resolved = _resolve_image_path(img_path, root)
            if resolved.exists():
                image_paths.append(resolved)

        if not image_paths:
            continue

        samples.append(
            {
                "title": build.get("title", ""),
                "build_url": build.get("build_url", ""),
                "image_paths": image_paths,
                "target": multi_hot,
                "true_tags": tags,
            }
        )

    return samples


def build_build_samples(dataset_json: Path, tag_to_idx: Dict[str, int], root: Path) -> List[Dict]:
    builds = _load_json_builds(dataset_json)
    return _expand_to_build_samples(builds, tag_to_idx, root)


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

    train_build_samples = _expand_to_build_samples(train_builds, tag_to_idx, root)
    val_build_samples = _expand_to_build_samples(val_builds, tag_to_idx, root)
    test_build_samples = _expand_to_build_samples(test_builds, tag_to_idx, root)

    if not train_build_samples:
        raise ValueError("No valid training build samples found.")

    return DatasetBundle(
        train_build_samples=train_build_samples,
        val_build_samples=val_build_samples,
        test_build_samples=test_build_samples,
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag,
    )
