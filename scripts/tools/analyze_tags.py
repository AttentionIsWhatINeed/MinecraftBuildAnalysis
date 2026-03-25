"""
Analyze split tags distribution from dataset.

This script helps you understand which split tags are most common,
to guide your decision on which tags to use in the filter configuration.

Workflow:
1. Load raw data and filter valid builds
2. Analyze split tags (tags split by space into words)
3. Display top N split tags with frequency
4. Help you decide which tags to add to config/filter_config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.process.data_processor import DataProcessor


def main():
    print("=" * 80)
    print("SPLIT TAGS ANALYSIS")
    print("=" * 80)
    
    # Step 1: Load and filter data
    print("\n[Step 1] Loading and filtering data...")
    processor = DataProcessor()
    
    valid_builds = processor.filter_valid_builds(
        require_tags=True,
        require_images=True,
        require_existing_files=False
    )
    
    print(f"✓ Loaded {len(valid_builds)} valid builds")
    
    # Step 2: Get basic statistics
    print("\n[Step 2] Dataset statistics...")
    
    stats = processor.get_statistics(valid_builds)
    top_tags = processor.get_top_tags(n=20, data=valid_builds)
    
    print(f"\nOriginal (non-split) tags:")
    print(f"  Total builds: {stats['total_builds']}")
    print(f"  Unique tags: {stats['unique_tags']}")
    print(f"  Avg tags/build: {stats['avg_tags_per_build']:.2f}")
    
    print(f"\nTop 20 original tags:")
    for i, (tag, count) in enumerate(top_tags, 1):
        print(f"  {i:2d}. {tag:40s} : {count:4d} occurrences")
    
    # Step 3: Analyze split tags
    print("\n[Step 3] Analyzing split tags...")
    
    split_tags_dist = processor.get_split_tags_distribution(data=valid_builds, n=100)
    
    print(f"\nTop 100 split tags (tags split by space into words):")
    print(f"{'#':<4} {'Tag':<50} {'Count':<8} {'% of builds':<12}")
    print("-" * 74)
    
    for i, (tag, count) in enumerate(split_tags_dist, 1):
        percentage = (count / len(valid_builds)) * 100
        print(f"{i:<4d} {tag:<50s} {count:<8d} {percentage:>6.1f}%")
    
    # Step 4: Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"""
Based on the analysis above:

1. REVIEW the top split tags list
   - These are the most common individual words in the dataset
   - Split from multi-word tags like "working mechanism" → "working", "mechanism"

2. CHOOSE which split tags you want to use:
   - Option A: Use top 10-15 tags (high quality, specific)
   - Option B: Use top 20-30 tags (balanced coverage)
   - Option C: Use top 50 tags (broad coverage, may include noise)
   - Option D: Custom selection based on your needs

3. EDIT the configuration file:
   - Open: config/filter_config.py
   - Add a new preset or modify existing one:
   
   Example:
   TAG_PRESETS["split_top_20"] = [
       "medieval",
       "statue",
       "working",
       "fantasy",
       ...  # Add top split tags here
   ]

4. GENERATE filtered dataset:
   - Edit scripts/generate_filtered_dataset.py
   - Change: preset_name = "split_top_20"
   - Run: python scripts/generate_filtered_dataset.py

5. VERIFY the results:
   - Check the statistics output
   - Compare with and without split_tags=True option
   - Adjust your tag selection as needed
""")
    
    # Step 5: Quick summary
    print("\n" + "=" * 80)
    print("QUICK STATS")
    print("=" * 80)
    
    print(f"\nSummary:")
    print(f"  Original unique tags: {stats['unique_tags']}")
    print(f"  Split tags analyzed: {len(split_tags_dist)}")
    print(f"  Top split tag: '{split_tags_dist[0][0]}' ({split_tags_dist[0][1]} occurrences)")
    
    top_10_tags = [tag for tag, _ in split_tags_dist[:10]]
    print(f"\n  Top 10 split tags for quick reference:")
    for i, tag in enumerate(top_10_tags, 1):
        print(f"    {i}. {tag}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Create your tag selection in config/filter_config.py
2. Update scripts/generate_filtered_dataset.py to use your preset
3. Run: python scripts/generate_filtered_dataset.py
4. Check data/processed/ for your filtered datasets

TIP: You can also use different tag combinations!
""")


if __name__ == "__main__":
    main()
