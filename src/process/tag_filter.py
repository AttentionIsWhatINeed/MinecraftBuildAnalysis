import logging
from typing import List, Dict, Set, Optional
from pathlib import Path
import json
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TagFilter:
    """
    Filter for selecting and processing builds based on specified tags.
    
    This class handles:
    - Filtering builds by tag list (keep only builds containing specified tags)
    - Keeping only specified tags in each build
    - Removing builds without any specified tags
    - Optional multi-word tag splitting (e.g., "working mechanism" → "working", "mechanism")
    """
    
    def __init__(self, tag_list: List[str], split_tags: bool = False):
        """
        Initialize Filter with a list of tags to keep.
        
        Args:
            tag_list: List of tags to keep in the dataset
            split_tags: If True, split multi-word tags by space
        """
        self.split_tags = split_tags
        
        # Apply tag splitting if enabled
        if self.split_tags:
            tag_list = self._split_tags(tag_list)

        tag_list = self._deduplicate_keep_order(tag_list)
        
        self.tag_set: Set[str] = set(tag_list)
        self.tag_list: List[str] = list(tag_list)
        logger.info(f"Filter initialized with {len(self.tag_set)} tags"
                   f"{' (with multi-word tag splitting)' if split_tags else ''}")

    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize build tags according to current splitting mode."""
        return self._split_tags(tags) if self.split_tags else tags

    def _deduplicate_keep_order(self, tags: List[str]) -> List[str]:
        """Remove duplicate tags while preserving the first-seen order."""
        return list(dict.fromkeys(tags))
    
    def _split_tags(self, tags: List[str]) -> List[str]:
        """
        Split multi-word tags into individual words.
        
        Args:
            tags: List of tags to split
        
        Returns:
            List of tags with multi-word tags split by space
        """
        split_result = []
        for tag in tags:
            if ' ' in tag:
                split_result.extend(tag.split())
            else:
                split_result.append(tag)
        
        return split_result
    
    def filter_builds_by_tags(
        self,
        builds: List[Dict],
        remove_tag_mismatch: bool = True
    ) -> List[Dict]:
        """
        Filter builds to keep only those with specified tags.
        
        Args:
            builds: List of builds to filter
            remove_tag_mismatch: If True, remove builds with no matching tags
        
        Returns:
            List of filtered builds (with only specified tags kept)
        """
        logger.info(f"Starting tag-based filtering on {len(builds)} builds...")
        
        filtered_builds = []
        removed_count = 0
        tags_kept_count = 0
        tags_removed_count = 0
        
        for build in builds:
            original_tags = self._normalize_tags(build.get('tags', []))
            
            # Keep only tags that are in the tag_set
            kept_tags = [tag for tag in original_tags if tag in self.tag_set]
            kept_tags = self._deduplicate_keep_order(kept_tags)
            
            # If no matching tags found
            if not kept_tags:
                if remove_tag_mismatch:
                    removed_count += 1
                    continue
            
            # Create a copy and update tags
            filtered_build = build.copy()
            filtered_build['tags'] = kept_tags
            filtered_builds.append(filtered_build)
            
            tags_kept_count += len(kept_tags)
            tags_removed_count += len(original_tags) - len(kept_tags)
        
        filtering_rate = (len(builds) - removed_count) / len(builds) * 100 if builds else 0
        logger.info(f"Filtering complete: {len(filtered_builds)}/{len(builds)} builds kept "
                   f"({filtering_rate:.1f}% retention)")
        logger.info(f"Tags: {tags_kept_count} kept, {tags_removed_count} removed from tag list")
        
        if removed_count > 0:
            logger.info(f"Builds without any specified tags: {removed_count} removed")
        
        return filtered_builds
    
    def get_tag_coverage(self, builds: List[Dict]) -> Dict:
        """
        Get statistics about tag coverage in the builds.
        
        Args:
            builds: List of builds to analyze
        
        Returns:
            Dictionary with tag coverage statistics
        """
        tag_counts = {}
        build_coverage = {}
        
        for build in builds:
            tags = self._normalize_tags(build.get('tags', []))
            for tag in tags:
                if tag in self.tag_set:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Count matching tags per build
            matching_count = sum(1 for tag in tags if tag in self.tag_set)
            if matching_count > 0:
                if matching_count not in build_coverage:
                    build_coverage[matching_count] = 0
                build_coverage[matching_count] += 1
        
        coverage = {
            'filter_tag_list_size': len(self.tag_set),
            'tags_found_in_builds': len(tag_counts),
            'tags_not_found': len(self.tag_set) - len(tag_counts),
            'tag_counts': tag_counts,
            'build_coverage': build_coverage,
            'builds_with_tags': sum(build_coverage.values()),
            'coverage_percentage': (len(tag_counts) / len(self.tag_set) * 100) if self.tag_set else 0
        }
        
        return coverage
    
    def get_missing_tags(self, builds: List[Dict]) -> List[str]:
        """
        Get tags from filter list that are not found in builds.
        
        Args:
            builds: List of builds to search
        
        Returns:
            List of tags not found in any build
        """
        found_tags = set()
        for build in builds:
            for tag in self._normalize_tags(build.get('tags', [])):
                if tag in self.tag_set:
                    found_tags.add(tag)
        
        missing = list(self.tag_set - found_tags)
        return sorted(missing)
    
    def preview_filtering(self, builds: List[Dict], n_samples: int = 3) -> Dict:
        """
        Preview what the filtering will do before applying.
        
        Args:
            builds: List of builds to preview
            n_samples: Number of example builds to show
        
        Returns:
            Dictionary with preview information
        """
        preview = {
            'filter_tags': self.tag_list,
            'total_builds': len(builds),
            'example_builds': []
        }
        
        # Show examples
        for i, build in enumerate(builds[:n_samples]):
            original_tags = self._normalize_tags(build.get('tags', []))
            kept_tags = [tag for tag in original_tags if tag in self.tag_set]
            
            preview['example_builds'].append({
                'title': build.get('title', 'Unknown'),
                'original_tags': original_tags,
                'tags_to_keep': kept_tags,
                'tags_to_remove': [tag for tag in original_tags if tag not in self.tag_set],
                'will_be_kept': len(kept_tags) > 0
            })
        
        return preview


    def create_train_val_test_split(
        self,
        builds: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: Optional[int] = None
    ) -> tuple:
        """
        Create train/validation/test splits from filtered builds.
        
        First applies tag filtering, then creates splits.
        
        Args:
            builds: List of builds to filter and split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Seed for reproducibility
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0 (got {ratio_sum:.6f})")

        # First filter the builds
        filtered_builds = self.filter_builds_by_tags(builds, remove_tag_mismatch=True)
        
        if not filtered_builds:
            logger.warning("No builds remain after tag filtering!")
            return [], [], []
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        
        # Shuffle data
        shuffled_data = filtered_builds.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices
        total = len(shuffled_data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Create splits
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        logger.info(f"Data split - Train: {len(train_data)} / "
                   f"Val: {len(val_data)} / Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_processed_dataset(
        self,
        output_file: str,
        data: List[Dict],
        include_stats: bool = True
    ) -> Path:
        """
        Save filtered dataset to JSON file.
        
        Args:
            output_file: Path where to save the dataset
            data: List of builds to save
            include_stats: If True, include statistics in the output
        
        Returns:
            Path to the saved file
        """
        if not data:
            logger.warning("No data to save!")
            return Path(output_file)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_builds': len(data),
                'filter_tags': self.tag_list,
                'total_images': sum(b.get('images_count', 0) for b in data),
            },
            'builds': data
        }
        
        if include_stats:
            # Calculate statistics
            all_tags = []
            image_counts = []
            
            for build in data:
                all_tags.extend(build.get('tags', []))
                image_counts.append(build.get('images_count', 0))
            
            stats = {
                'total_builds': len(data),
                'total_images': sum(image_counts),
                'avg_images_per_build': sum(image_counts) / len(data) if data else 0,
                'unique_tags': len(set(all_tags)),
                'total_tag_occurrences': len(all_tags),
                'avg_tags_per_build': len(all_tags) / len(data) if data else 0,
            }
            
            output['statistics'] = stats
            
            # Calculate tag frequencies
            tag_counts = {}
            for build in data:
                for tag in build.get('tags', []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            output['tag_distribution'] = [
                {'tag': tag, 'count': count}
                for tag, count in sorted_tags
            ]
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved filtered dataset with {len(data)} builds to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

