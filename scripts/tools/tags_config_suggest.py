"""
Configure filter tags based on split tags analysis.

This script helps you create a new filter preset in config/filter_config.py
based on the split tags analysis results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.process.data_processor import DataProcessor


def main():
    print("=" * 80)
    print("CONFIGURE FILTER TAGS")
    print("=" * 80)
    
    # Load data
    processor = DataProcessor()
    valid_builds = processor.filter_valid_builds(
        require_tags=True,
        require_images=True,
        require_existing_files=False
    )
    
    # Get split tags
    split_tags_dist = processor.get_split_tags_distribution(
        data=valid_builds, 
        n=100
    )
    
    print(f"\nAvailable options to add to config/filter_config.py:")
    print(f"\n1. Top 10 split tags:")
    top_10 = [tag for tag, _ in split_tags_dist[:10]]
    print(f"   TAG_PRESETS['split_top_10'] = {top_10}")
    
    print(f"\n2. Top 20 split tags:")
    top_20 = [tag for tag, _ in split_tags_dist[:20]]
    print(f"   TAG_PRESETS['split_top_20'] = {top_20}")
    
    print(f"\n3. Top 30 split tags:")
    top_30 = [tag for tag, _ in split_tags_dist[:30]]
    print(f"   TAG_PRESETS['split_top_30'] = {top_30}")
    
    print(f"\n4. Top 50 split tags:")
    top_50 = [tag for tag, _ in split_tags_dist[:50]]
    print(f"   TAG_PRESETS['split_top_50'] = {top_50}")
    
    print(f"\n" + "=" * 80)
    print("INSTRUCTIONS")
    print("=" * 80)
    print(f"""
1. Open: config/filter_config.py

2. Add one of the presets above to TAG_PRESETS dictionary:
   
   Example (using top 20):
   TAG_PRESETS = {{
       ...existing presets...
       
       "split_top_20": {top_20},
   }}

3. Save the file

4. Update scripts/generate_filtered_dataset.py:
   preset_name = "split_top_20"  # Change this

5. Run: python scripts/generate_filtered_dataset.py

6. Find your results in: data/processed/
""")


if __name__ == "__main__":
    main()
