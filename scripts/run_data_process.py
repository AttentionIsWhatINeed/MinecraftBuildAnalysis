"""
Generate filtered dataset with tag-based filtering and splits.

Workflow:
1. DataProcessor: Load raw data, filter valid builds, and deduplicate builds
2. DataProcessor: Display top tags as reference information
3. TagFilter: Apply user-specified tag filtering
4. TagFilter: Create train/val/test splits
5. TagFilter: Save datasets

To change which tags are used:
- Option 1: Edit the filter_config.py presets
- Option 2: Modify the 'filter_tags' list below
"""

import sys
import argparse
import json
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.process.data_processor import DataProcessor
from src.process.tag_filter import TagFilter
from config.filter_config import get_tags


@dataclass
class GenerateConfig:
    """Runtime configuration for dataset generation."""

    preset: str = "custom"
    label_mode: str = "auto"
    manual_vocab_file: str | None = None
    split_tags: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    metadata_file: str = "data/raw/metadata/builds_metadata.json"
    output_dir: str = "data/processed"
    max_labels: int = 24
    min_tag_builds: int = 8
    max_tag_ratio: float = 0.40
    max_pair_jaccard: float = 0.80
    vocab_report_file: str | None = None


def _dedupe_keep_order(tags: List[str]) -> List[str]:
    return list(dict.fromkeys(tags))


def _normalize_tags(tags: List[str], split_tags: bool) -> List[str]:
    normalized: List[str] = []
    for raw in tags:
        tag = str(raw).strip()
        if not tag:
            continue

        if split_tags and " " in tag:
            normalized.extend([w for w in tag.split() if w])
        else:
            normalized.append(tag)

    return _dedupe_keep_order(normalized)


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(union, 1)


def _build_auto_label_vocab(
    builds: List[Dict],
    split_tags: bool,
    max_labels: int,
    min_tag_builds: int,
    max_tag_ratio: float,
    max_pair_jaccard: float,
) -> tuple[List[str], Dict]:
    total_builds = len(builds)
    if total_builds == 0:
        return [], {
            "total_builds": 0,
            "selected_tags": [],
        }

    tag_to_builds: Dict[str, Set[int]] = defaultdict(set)
    for build_idx, build in enumerate(builds):
        tags = _normalize_tags(build.get("tags", []), split_tags=split_tags)
        for tag in tags:
            tag_to_builds[tag].add(build_idx)

    rows = []
    low_support = []
    too_generic = []

    for tag, build_ids in tag_to_builds.items():
        count = len(build_ids)
        ratio = count / total_builds
        row = {
            "tag": tag,
            "build_count": count,
            "build_ratio": ratio,
        }
        rows.append(row)

        if count < min_tag_builds:
            low_support.append(tag)
        elif ratio > max_tag_ratio:
            too_generic.append(tag)

    candidate_rows = [
        r
        for r in rows
        if r["build_count"] >= min_tag_builds and r["build_ratio"] <= max_tag_ratio
    ]
    candidate_rows.sort(key=lambda r: (-r["build_count"], r["tag"]))

    selected: List[str] = []
    selected_sets: Dict[str, Set[int]] = {}
    covered_builds: Set[int] = set()
    remaining = candidate_rows.copy()

    while remaining and len(selected) < max_labels:
        best_idx = -1
        best_gain = -1
        best_count = -1

        for i, row in enumerate(remaining):
            tag = row["tag"]
            tag_set = tag_to_builds[tag]

            max_overlap = 0.0
            for selected_tag, selected_set in selected_sets.items():
                overlap = _jaccard(tag_set, selected_set)
                if overlap > max_overlap:
                    max_overlap = overlap

            if max_overlap > max_pair_jaccard:
                continue

            gain = len(tag_set - covered_builds)
            count = row["build_count"]

            if gain > best_gain or (gain == best_gain and count > best_count):
                best_gain = gain
                best_count = count
                best_idx = i

        if best_idx < 0:
            break

        chosen = remaining.pop(best_idx)
        chosen_tag = chosen["tag"]
        selected.append(chosen_tag)
        selected_sets[chosen_tag] = tag_to_builds[chosen_tag]
        covered_builds |= tag_to_builds[chosen_tag]

    if not selected and candidate_rows:
        selected = [row["tag"] for row in candidate_rows[: max_labels]]
        for tag in selected:
            covered_builds |= tag_to_builds[tag]

    report = {
        "total_builds": total_builds,
        "split_tags": split_tags,
        "selection_params": {
            "max_labels": max_labels,
            "min_tag_builds": min_tag_builds,
            "max_tag_ratio": max_tag_ratio,
            "max_pair_jaccard": max_pair_jaccard,
        },
        "candidate_tags_total": len(rows),
        "candidates_after_filter": len(candidate_rows),
        "selected_count": len(selected),
        "selected_tags": selected,
        "retained_builds_by_selected_tags": len(covered_builds),
        "retained_build_ratio": len(covered_builds) / total_builds,
        "filtered_out": {
            "low_support_count": len(low_support),
            "too_generic_count": len(too_generic),
            "low_support_examples": sorted(low_support)[:30],
            "too_generic_examples": sorted(too_generic)[:30],
        },
        "top_candidates": candidate_rows[:100],
    }
    return selected, report


def _load_manual_vocab(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Manual vocab file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return _dedupe_keep_order([str(t).strip() for t in obj if str(t).strip()])

    if isinstance(obj, dict):
        tags = obj.get("selected_tags", obj.get("tags", []))
        if isinstance(tags, list):
            return _dedupe_keep_order([str(t).strip() for t in tags if str(t).strip()])

    raise ValueError(
        "Manual vocab JSON must be a list of tags or an object with selected_tags/tags list."
    )


def _resolve_filter_tags(config: GenerateConfig, valid_builds: List[Dict], output_dir: Path) -> tuple[List[str], Dict]:
    if config.label_mode == "preset":
        tags = get_tags(config.preset)
        return tags, {
            "label_mode": "preset",
            "selected_tags": tags,
            "preset": config.preset,
        }

    if config.label_mode == "manual":
        if config.manual_vocab_file is None:
            raise ValueError("--manual-vocab-file is required when --label-mode manual")
        path = Path(config.manual_vocab_file)
        tags = _load_manual_vocab(path)
        return tags, {
            "label_mode": "manual",
            "selected_tags": tags,
            "manual_vocab_file": str(path),
        }

    tags, report = _build_auto_label_vocab(
        builds=valid_builds,
        split_tags=config.split_tags,
        max_labels=config.max_labels,
        min_tag_builds=config.min_tag_builds,
        max_tag_ratio=config.max_tag_ratio,
        max_pair_jaccard=config.max_pair_jaccard,
    )

    report["label_mode"] = "auto"

    report_path = Path(config.vocab_report_file) if config.vocab_report_file else (output_dir / "auto_label_vocab_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report["report_file"] = str(report_path)
    return tags, report


def parse_args() -> argparse.Namespace:
    """Parse command-line args for dataset processing entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run dataset processing with configurable preset and split options."
    )
    parser.add_argument("--preset", default="custom", help="Tag preset name")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="auto",
        choices=["auto", "preset", "manual"],
        help="How to build final label dictionary.",
    )
    parser.add_argument(
        "--manual-vocab-file",
        type=str,
        default=None,
        help="Path to JSON tag dictionary used when --label-mode manual.",
    )
    parser.add_argument(
        "--split-tags",
        action="store_true",
        default=True,
        help="Enable splitting of multi-word tags (default: enabled)",
    )
    parser.add_argument(
        "--no-split-tags",
        action="store_false",
        dest="split_tags",
        help="Disable splitting of multi-word tags",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--metadata-file",
        default="data/raw/metadata/builds_metadata.json",
        help="Input metadata file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for generated JSON files",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=24,
        help="Maximum number of labels when --label-mode auto.",
    )
    parser.add_argument(
        "--min-tag-builds",
        type=int,
        default=8,
        help="Minimum build count for a tag to be considered in auto mode.",
    )
    parser.add_argument(
        "--max-tag-ratio",
        type=float,
        default=0.40,
        help="Drop too-generic tags appearing in more than this ratio of builds.",
    )
    parser.add_argument(
        "--max-pair-jaccard",
        type=float,
        default=0.80,
        help="Avoid selecting near-duplicate tags with jaccard overlap above this value.",
    )
    parser.add_argument(
        "--vocab-report-file",
        type=str,
        default=None,
        help="Where to save auto-selected vocabulary report JSON.",
    )
    return parser.parse_args()


def run_from_cli() -> None:
    """CLI entrypoint that maps args into runtime config."""
    args = parse_args()
    config = GenerateConfig(
        preset=args.preset,
        label_mode=args.label_mode,
        manual_vocab_file=args.manual_vocab_file,
        split_tags=args.split_tags,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        max_labels=args.max_labels,
        min_tag_builds=args.min_tag_builds,
        max_tag_ratio=args.max_tag_ratio,
        max_pair_jaccard=args.max_pair_jaccard,
        vocab_report_file=args.vocab_report_file,
    )
    main(config)


def main(config: GenerateConfig | None = None) -> None:
    """Generate filtered dataset with specified configuration."""
    if config is None:
        config = GenerateConfig()

    ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0 (got {ratio_sum:.6f})")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FILTERED DATASET GENERATION")
    print("=" * 80)
    
    # ==============================================
    # PHASE 1: DATA PROCESSOR (cleaning + stats)
    # ==============================================
    
    print("\n[PHASE 1: DATA PROCESSING]")
    print("-" * 80)
    
    # Step 1: Initialize processor and clean data
    print("\n[Step 1] Loading raw metadata and filtering valid builds...")
    processor = DataProcessor(metadata_file=config.metadata_file)
    
    valid_builds = processor.filter_valid_builds(
        require_tags=True,
        require_images=True,
        require_existing_files=False,
    )

    filter_report = processor.last_filter_report
    print(
        f"✓ Loaded {len(valid_builds)} valid builds "
        f"(removed builds without tags/images and deduplicated by identity)"
    )
    print(
        f"  Deduplication: {filter_report.get('valid_before_dedup', len(valid_builds))} "
        f"→ {filter_report.get('valid_after_dedup', len(valid_builds))} "
        f"(removed {filter_report.get('duplicates_removed', 0)})"
    )
    
    # Step 2: Display top tags as reference information
    print("\n[Step 2] Top tags reference information...")
    
    top_tags = processor.get_top_tags(n=20, data=valid_builds)
    print(f"Top 20 tags in dataset (for reference):")
    for i, (tag, count) in enumerate(top_tags, 1):
        print(f"  {i:2d}. {tag:30s} : {count:4d} occurrences")
    
    # Step 3: Display tag distribution
    print("\n[Step 3] Original dataset statistics...")
    
    stats_original = processor.get_statistics(valid_builds)
    print(f"Dataset overview:")
    print(f"  Total builds: {stats_original['total_builds']}")
    print(f"  Unique tags: {stats_original['unique_tags']}")
    print(f"  Avg tags/build: {stats_original['avg_tags_per_build']:.2f}")
    
    # ==============================================
    # PHASE 2: TAG FILTER (filtering + split)
    # ==============================================
    
    print("\n" + "=" * 80)
    print("[PHASE 2: TAG-BASED FILTERING & SPLITTING]")
    print("-" * 80)
    
    # Step 4: Build/select label dictionary
    print("\n[Step 4] Building label dictionary...")

    enable_tag_splitting = config.split_tags
    filter_tags, vocab_report = _resolve_filter_tags(config, valid_builds, output_dir)

    if not filter_tags:
        raise ValueError("No tags selected for filtering. Adjust label-mode settings.")

    print(f"Label mode: {config.label_mode}")
    if config.label_mode == "preset":
        print(f"Preset: {config.preset}")
    if config.label_mode == "manual":
        print(f"Manual vocab file: {config.manual_vocab_file}")
    if config.label_mode == "auto":
        print(f"Auto-selected labels: {len(filter_tags)}")
        print(
            "Coverage by selected labels: "
            f"{vocab_report.get('retained_builds_by_selected_tags', 0)}/"
            f"{vocab_report.get('total_builds', 0)} "
            f"({100 * vocab_report.get('retained_build_ratio', 0.0):.1f}%)"
        )
        if "report_file" in vocab_report:
            print(f"Saved vocab report: {vocab_report['report_file']}")
            print("You can manually edit selected_tags in this JSON and rerun with --label-mode manual.")

    if enable_tag_splitting:
        print("✓ Multi-word tag splitting enabled")
        print("  Example: 'working mechanism' -> 'working', 'mechanism'")

    print("Final tags to filter:")
    for i, tag in enumerate(filter_tags, 1):
        print(f"  {i:2d}. {tag}")
    
    # Step 5: Create tag filter
    print("\n[Step 5] Creating tag filter...")
    
    tag_filter = TagFilter(filter_tags, split_tags=enable_tag_splitting)
    
    # Step 6: Preview filtering
    print("\n[Step 6] Previewing tag filter effect...")
    
    preview = tag_filter.preview_filtering(valid_builds, n_samples=3)
    print(f"Example transformations:")
    for i, example in enumerate(preview['example_builds'], 1):
        print(f"\n  {i}. {example['title']}")
        if example['tags_to_keep']:
            print(f"     Keep: {example['tags_to_keep']}")
        else:
            print(f"     REMOVED (no matching tags)")
    
    # Step 7: Check coverage
    print("\n[Step 7] Checking tag coverage...")
    
    coverage = tag_filter.get_tag_coverage(valid_builds)
    print(f"Filter tags found: {coverage['tags_found_in_builds']}/{coverage['filter_tag_list_size']}")
    print(f"Builds with these tags: {coverage['builds_with_tags']}")
    print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
    
    # Step 8: Apply filter and create splits
    print("\n[Step 8] Creating train/val/test splits with tag filtering...")
    
    train, val, test = tag_filter.create_train_val_test_split(
        builds=valid_builds,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.seed,
    )
    
    print(f"Dataset splits:")
    print(f"  Training: {len(train)} builds (70%)")
    print(f"  Validation: {len(val)} builds (15%)")
    print(f"  Test: {len(test)} builds (15%)")
    print(f"  Total: {len(train) + len(val) + len(test)} builds")
    
    # Step 9: Save filtered datasets
    print("\n[Step 9] Saving filtered datasets...")
    
    output_file = output_dir / "minecraft_builds_filtered.json"
    train_file = output_dir / "minecraft_builds_filtered_train.json"
    val_file = output_dir / "minecraft_builds_filtered_val.json"
    test_file = output_dir / "minecraft_builds_filtered_test.json"
    
    # Save combined
    all_filtered = train + val + test
    tag_filter.save_processed_dataset(
        str(output_file),
        data=all_filtered,
        include_stats=True
    )
    print(f"✓ Full dataset: {output_file}")
    
    # Save splits
    tag_filter.save_processed_dataset(str(train_file), train, include_stats=True)
    tag_filter.save_processed_dataset(str(val_file), val, include_stats=True)
    tag_filter.save_processed_dataset(str(test_file), test, include_stats=True)
    
    print(f"✓ Training set: {train_file}")
    print(f"✓ Validation set: {val_file}")
    print(f"✓ Test set: {test_file}")
    
    # Step 10: Statistics comparison
    print("\n[Step 10] Statistics comparison...")
    
    stats_filtered = processor.get_statistics(all_filtered)
    
    print(f"\nOriginal Dataset:")
    print(f"  Builds: {stats_original['total_builds']}")
    print(f"  Unique tags: {stats_original['unique_tags']}")
    print(f"  Avg tags/build: {stats_original['avg_tags_per_build']:.2f}")
    
    print(f"\nFiltered Dataset (with user-specified tags):")
    print(f"  Builds: {stats_filtered['total_builds']}")
    print(f"  Unique tags: {stats_filtered['unique_tags']}")
    print(f"  Avg tags/build: {stats_filtered['avg_tags_per_build']:.2f}")
    
    print(f"\nChange:")
    print(f"  Builds: {stats_original['total_builds']} → {stats_filtered['total_builds']} "
          f"({stats_filtered['total_builds']/stats_original['total_builds']*100:.1f}%)")
    print(f"  Unique tags: {stats_original['unique_tags']} → {stats_filtered['unique_tags']}")
    
    print("\n" + "=" * 80)
    print("FILTERED DATASET GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_from_cli()
