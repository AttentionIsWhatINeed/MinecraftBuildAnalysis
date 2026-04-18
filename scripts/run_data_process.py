"""
Generate filtered dataset with blacklist-based tag filtering and splits.

Workflow:
1. DataProcessor: Load raw data, filter valid builds, and deduplicate builds
2. DataProcessor: Display top tags as reference information
3. TagFilter: Remove tags configured in blacklist
4. TagFilter: Create train/val/test splits
5. TagFilter: Save datasets

To change blacklist behavior, edit BLACKLIST_TAGS in config/filter_config.py.
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.process.data_processor import DataProcessor
from src.process.tag_filter import TagFilter
from config.filter_config import get_blacklist_tags


@dataclass
class GenerateConfig:
    """Runtime configuration for dataset generation."""

    split_tags: bool = False
    min_tag_occurrences: int = 10
    top_k_tags: int = 30
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 13
    metadata_file: str = "data/raw/metadata/builds_metadata.json"
    output_dir: str = "data/processed"

def parse_args() -> argparse.Namespace:
    """Parse command-line args for dataset processing entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run dataset processing with blacklist-based tag filtering."
    )
    parser.add_argument(
        "--split-tags",
        action="store_true",
        default=False,
        help="Enable splitting of multi-word tags (default: disabled)",
    )
    parser.add_argument(
        "--min-tag-occurrences",
        type=int,
        default=1,
        help="Drop tags that appear fewer times than this threshold after blacklist filtering.",
    )
    parser.add_argument(
        "--top-k-tags",
        type=int,
        default=30,
        help="Automatically keep only top-k frequent tags after blacklist/threshold filtering. Use 0 to keep all.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
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
    return parser.parse_args()


def run_from_cli() -> None:
    """CLI entrypoint that maps args into runtime config."""
    args = parse_args()
    config = GenerateConfig(
        split_tags=args.split_tags,
        min_tag_occurrences=args.min_tag_occurrences,
        top_k_tags=args.top_k_tags,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
    )
    main(config)


def main(config: GenerateConfig | None = None) -> None:
    """Generate filtered dataset with specified configuration."""
    if config is None:
        config = GenerateConfig()

    if config.min_tag_occurrences < 1:
        raise ValueError("min_tag_occurrences must be >= 1")
    if config.top_k_tags < 0:
        raise ValueError("top_k_tags must be >= 0")

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
    
    # Step 4: Load blacklist configuration
    print("\n[Step 4] Loading blacklist configuration...")

    enable_tag_splitting = config.split_tags
    blacklist_tags = get_blacklist_tags()

    print(f"Blacklist tags configured: {len(blacklist_tags)}")
    if blacklist_tags:
        for i, tag in enumerate(blacklist_tags, 1):
            print(f"  {i:2d}. {tag}")
    else:
        print("  (empty blacklist, no tags will be removed)")

    if enable_tag_splitting:
        print("✓ Multi-word tag splitting enabled")
        print("  Example: 'working mechanism' -> 'working', 'mechanism'")
    print(f"Tag min occurrence threshold: {config.min_tag_occurrences}")
    if config.top_k_tags > 0:
        print(f"Auto keep top-k tags: {config.top_k_tags}")
    else:
        print("Auto keep top-k tags: disabled (keep all tags)")
    
    # Step 5: Create tag filter
    print("\n[Step 5] Creating tag filter...")
    
    tag_filter = TagFilter(
        blacklist_tags,
        split_tags=enable_tag_splitting,
        min_tag_occurrences=config.min_tag_occurrences,
        top_k_tags=(config.top_k_tags if config.top_k_tags > 0 else None),
    )
    
    # Step 6: Preview filtering
    print("\n[Step 6] Previewing tag filter effect...")
    
    preview = tag_filter.preview_filtering(valid_builds, n_samples=3)
    print(f"Example transformations:")
    for i, example in enumerate(preview['example_builds'], 1):
        print(f"\n  {i}. {example['title']}")
        if example['removed_by_blacklist']:
            print(f"     Remove by blacklist: {example['removed_by_blacklist']}")
        if example['removed_by_threshold']:
            print(f"     Remove by threshold: {example['removed_by_threshold']}")
        if example['removed_by_top_k']:
            print(f"     Remove by top-k: {example['removed_by_top_k']}")
        if example['tags_to_remove']:
            print(f"     Remove: {example['tags_to_remove']}")
        if example['tags_to_keep']:
            print(f"     Keep: {example['tags_to_keep']}")
        else:
            print(f"     REMOVED (no tags remain after filtering)")
    
    # Step 7: Check coverage
    print("\n[Step 7] Checking blacklist coverage...")
    
    coverage = tag_filter.get_tag_coverage(valid_builds)
    print(
        "Blacklist tags found in dataset: "
        f"{coverage['blacklisted_tags_found_in_builds']}/{coverage['blacklist_size']}"
    )
    print(f"Builds containing blacklist tags: {coverage['builds_with_blacklisted_tags']}")
    print(f"Blacklist coverage: {coverage['coverage_percentage']:.1f}%")
    if config.min_tag_occurrences > 1:
        print(
            "Low-frequency tags to drop: "
            f"{coverage['low_frequency_tag_count']} "
            f"(threshold < {config.min_tag_occurrences})"
        )
    if coverage['top_k_requested'] is not None:
        print(
            "Top-k tags selected: "
            f"{coverage['top_k_selected_tag_count']}/"
            f"{coverage['threshold_kept_tag_count']} "
            f"(k={coverage['top_k_requested']})"
        )
        preview_top_tags = coverage['top_k_selected_tags'][:20]
        if preview_top_tags:
            print("Top-k selected tag preview:")
            for idx, tag in enumerate(preview_top_tags, 1):
                print(f"  {idx:2d}. {tag}")
    
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
    
    print(f"\nFiltered Dataset (blacklist applied):")
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
