import logging
from typing import List, Dict, Set, Optional
from pathlib import Path
import json
from datetime import datetime
import random
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TagFilter:
    """
    Filter for processing builds with blacklist-based tag removal.

    This class handles:
    - Removing blacklisted tags from each build
    - Dropping low-frequency tags based on minimum occurrence threshold
    - Optionally splitting multi-word tags (e.g., "working mechanism" -> "working", "mechanism")
    - Removing builds that end up with no remaining tags
    """
    
    def __init__(
        self,
        blacklist_tags: List[str],
        split_tags: bool = False,
        min_tag_occurrences: int = 1,
        top_k_tags: Optional[int] = None,
    ):
        """
        Initialize filter with blacklist tags.
        
        Args:
            blacklist_tags: List of tags to remove from the dataset
            split_tags: If True, split multi-word tags by space
            min_tag_occurrences: Minimum occurrences required to keep a tag
            top_k_tags: Keep only the top-k most frequent tags after threshold filtering
        """
        self.split_tags = split_tags

        if min_tag_occurrences < 1:
            raise ValueError("min_tag_occurrences must be >= 1")
        self.min_tag_occurrences = min_tag_occurrences

        if top_k_tags is not None and top_k_tags < 1:
            raise ValueError("top_k_tags must be >= 1 when provided")
        self.top_k_tags = top_k_tags

        if self.split_tags:
            blacklist_tags = self._split_tags(blacklist_tags)

        blacklist_tags = self._deduplicate_keep_order(blacklist_tags)

        self.blacklist_set: Set[str] = set(blacklist_tags)
        self.blacklist_tags: List[str] = list(blacklist_tags)
        self.last_filter_report: Dict = {}
        logger.info(f"Filter initialized with {len(self.blacklist_set)} blacklist tags"
                   f", min_tag_occurrences={self.min_tag_occurrences}"
               f", top_k_tags={self.top_k_tags if self.top_k_tags is not None else 'all'}"
                   f"{' (with multi-word tag splitting)' if split_tags else ''}")

    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize build tags according to current splitting mode."""
        normalized: List[str] = []
        for raw in tags:
            tag = str(raw).strip()
            if not tag:
                continue

            if self.split_tags and ' ' in tag:
                normalized.extend([word for word in tag.split() if word])
            else:
                normalized.append(tag)

        return self._deduplicate_keep_order(normalized)

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
        split_result: List[str] = []
        for raw in tags:
            tag = str(raw).strip()
            if not tag:
                continue
            if ' ' in tag:
                split_result.extend([word for word in tag.split() if word])
            else:
                split_result.append(tag)
        
        return split_result

    def _apply_min_occurrence_threshold(
        self,
        builds: List[Dict],
        remove_tag_mismatch: bool,
    ) -> tuple[List[Dict], Dict]:
        """Drop tags whose global occurrences are below configured threshold."""
        stats = {
            'min_tag_occurrences': self.min_tag_occurrences,
            'dropped_tag_count': 0,
            'dropped_tags': [],
            'tags_removed_by_threshold': 0,
            'builds_removed_by_threshold': 0,
        }

        if self.min_tag_occurrences <= 1 or not builds:
            return builds, stats

        tag_counts: Counter[str] = Counter()
        for build in builds:
            tag_counts.update(build.get('tags', []))

        dropped_tags = {
            tag
            for tag, count in tag_counts.items()
            if count < self.min_tag_occurrences
        }
        if not dropped_tags:
            return builds, stats

        threshold_filtered_builds = []
        tags_removed_by_threshold = 0
        builds_removed_by_threshold = 0

        for build in builds:
            original_tags = build.get('tags', [])
            kept_tags = [tag for tag in original_tags if tag not in dropped_tags]
            tags_removed_by_threshold += len(original_tags) - len(kept_tags)

            if not kept_tags and remove_tag_mismatch:
                builds_removed_by_threshold += 1
                continue

            filtered_build = build.copy()
            filtered_build['tags'] = kept_tags
            threshold_filtered_builds.append(filtered_build)

        stats.update({
            'dropped_tag_count': len(dropped_tags),
            'dropped_tags': sorted(dropped_tags),
            'tags_removed_by_threshold': tags_removed_by_threshold,
            'builds_removed_by_threshold': builds_removed_by_threshold,
        })
        return threshold_filtered_builds, stats

    def _apply_top_k_tags(
        self,
        builds: List[Dict],
        remove_tag_mismatch: bool,
    ) -> tuple[List[Dict], Dict]:
        """Keep only top-k frequent tags from threshold-filtered builds."""
        stats = {
            'top_k_requested': self.top_k_tags,
            'selected_top_tag_count': 0,
            'selected_top_tags': [],
            'tags_removed_by_top_k': 0,
            'builds_removed_by_top_k': 0,
        }

        if self.top_k_tags is None or not builds:
            return builds, stats

        tag_counts: Counter[str] = Counter()
        for build in builds:
            tag_counts.update(build.get('tags', []))

        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
        selected_top_tags = [tag for tag, _ in sorted_tags[: self.top_k_tags]]
        selected_top_set = set(selected_top_tags)

        top_k_filtered_builds = []
        tags_removed_by_top_k = 0
        builds_removed_by_top_k = 0

        for build in builds:
            original_tags = build.get('tags', [])
            kept_tags = [tag for tag in original_tags if tag in selected_top_set]
            tags_removed_by_top_k += len(original_tags) - len(kept_tags)

            if not kept_tags and remove_tag_mismatch:
                builds_removed_by_top_k += 1
                continue

            filtered_build = build.copy()
            filtered_build['tags'] = kept_tags
            top_k_filtered_builds.append(filtered_build)

        stats.update({
            'selected_top_tag_count': len(selected_top_tags),
            'selected_top_tags': selected_top_tags,
            'tags_removed_by_top_k': tags_removed_by_top_k,
            'builds_removed_by_top_k': builds_removed_by_top_k,
        })
        return top_k_filtered_builds, stats
    
    def filter_builds_by_tags(
        self,
        builds: List[Dict],
        remove_tag_mismatch: bool = True
    ) -> List[Dict]:
        """
        Filter builds by removing blacklisted tags.
        
        Args:
            builds: List of builds to filter
            remove_tag_mismatch: If True, remove builds with no remaining tags
        
        Returns:
            List of filtered builds (with blacklisted tags removed)
        """
        logger.info(f"Starting blacklist-based filtering on {len(builds)} builds...")
        
        filtered_builds = []
        removed_count_by_blacklist = 0
        kept_tag_total = 0
        removed_tag_total_by_blacklist = 0
        
        for build in builds:
            original_tags = self._normalize_tags(build.get('tags', []))

            kept_tags = [tag for tag in original_tags if tag not in self.blacklist_set]
            kept_tags = self._deduplicate_keep_order(kept_tags)

            if not kept_tags:
                if remove_tag_mismatch:
                    removed_count_by_blacklist += 1
                    continue

            filtered_build = build.copy()
            filtered_build['tags'] = kept_tags
            filtered_builds.append(filtered_build)

            kept_tag_total += len(kept_tags)
            removed_tag_total_by_blacklist += len(original_tags) - len(kept_tags)

        filtered_builds, threshold_stats = self._apply_min_occurrence_threshold(
            filtered_builds,
            remove_tag_mismatch=remove_tag_mismatch,
        )

        filtered_builds, top_k_stats = self._apply_top_k_tags(
            filtered_builds,
            remove_tag_mismatch=remove_tag_mismatch,
        )

        removed_count_total = (
            removed_count_by_blacklist
            + threshold_stats['builds_removed_by_threshold']
            + top_k_stats['builds_removed_by_top_k']
        )

        removed_tag_total = (
            removed_tag_total_by_blacklist
            + threshold_stats['tags_removed_by_threshold']
            + top_k_stats['tags_removed_by_top_k']
        )
        
        filtering_rate = (len(builds) - removed_count_total) / len(builds) * 100 if builds else 0
        logger.info(f"Filtering complete: {len(filtered_builds)}/{len(builds)} builds kept "
                   f"({filtering_rate:.1f}% retention)")
        logger.info(
            f"Tags: {kept_tag_total} kept, {removed_tag_total} removed "
            f"(blacklist={removed_tag_total_by_blacklist}, "
            f"threshold={threshold_stats['tags_removed_by_threshold']}, "
            f"top_k={top_k_stats['tags_removed_by_top_k']})"
        )

        if threshold_stats['dropped_tag_count'] > 0:
            logger.info(
                f"Low-frequency threshold applied (min occurrences={self.min_tag_occurrences}): "
                f"dropped {threshold_stats['dropped_tag_count']} tags"
            )

        if self.top_k_tags is not None:
            logger.info(
                f"Top-k selection applied: keep {top_k_stats['selected_top_tag_count']} tags "
                f"(requested top_k={self.top_k_tags})"
            )
        
        if removed_count_total > 0:
            logger.info(
                "Builds removed due to empty tags: "
                f"{removed_count_total} (blacklist={removed_count_by_blacklist}, "
                f"threshold={threshold_stats['builds_removed_by_threshold']}, "
                f"top_k={top_k_stats['builds_removed_by_top_k']})"
            )

        self.last_filter_report = {
            'input_builds': len(builds),
            'output_builds': len(filtered_builds),
            'removed_builds_by_blacklist': removed_count_by_blacklist,
            'removed_builds_by_threshold': threshold_stats['builds_removed_by_threshold'],
            'removed_builds_by_top_k': top_k_stats['builds_removed_by_top_k'],
            'removed_tags_by_blacklist': removed_tag_total_by_blacklist,
            'removed_tags_by_threshold': threshold_stats['tags_removed_by_threshold'],
            'removed_tags_by_top_k': top_k_stats['tags_removed_by_top_k'],
            'dropped_tags_by_threshold': threshold_stats['dropped_tags'],
            'selected_top_tags': top_k_stats['selected_top_tags'],
            'top_k_requested': self.top_k_tags,
            'min_tag_occurrences': self.min_tag_occurrences,
        }
        
        return filtered_builds
    
    def get_tag_coverage(self, builds: List[Dict]) -> Dict:
        """
        Get statistics about blacklist coverage in the builds.
        
        Args:
            builds: List of builds to analyze
        
        Returns:
            Dictionary with blacklist coverage statistics
        """
        tag_counts = {}
        build_coverage = {}
        
        for build in builds:
            tags = self._normalize_tags(build.get('tags', []))
            for tag in tags:
                if tag in self.blacklist_set:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            matching_count = sum(1 for tag in tags if tag in self.blacklist_set)
            if matching_count > 0:
                if matching_count not in build_coverage:
                    build_coverage[matching_count] = 0
                build_coverage[matching_count] += 1
        
        coverage = {
            'blacklist_size': len(self.blacklist_set),
            'blacklisted_tags_found_in_builds': len(tag_counts),
            'blacklisted_tags_not_found': len(self.blacklist_set) - len(tag_counts),
            'blacklisted_tag_counts': tag_counts,
            'build_coverage': build_coverage,
            'builds_with_blacklisted_tags': sum(build_coverage.values()),
            'coverage_percentage': (len(tag_counts) / len(self.blacklist_set) * 100) if self.blacklist_set else 0,
            'min_tag_occurrences': self.min_tag_occurrences,
            'top_k_requested': self.top_k_tags,
        }

        post_blacklist_counts: Counter[str] = Counter()
        for build in builds:
            normalized_tags = self._normalize_tags(build.get('tags', []))
            kept_tags = [tag for tag in normalized_tags if tag not in self.blacklist_set]
            post_blacklist_counts.update(kept_tags)

        low_frequency_tag_counts = {
            tag: count
            for tag, count in post_blacklist_counts.items()
            if count < self.min_tag_occurrences
        }
        coverage['low_frequency_tag_count'] = len(low_frequency_tag_counts)
        coverage['low_frequency_tag_counts'] = dict(
            sorted(low_frequency_tag_counts.items(), key=lambda x: (x[1], x[0]))
        )

        threshold_kept_counts = {
            tag: count
            for tag, count in post_blacklist_counts.items()
            if count >= self.min_tag_occurrences
        }
        sorted_threshold_kept = sorted(
            threshold_kept_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )

        if self.top_k_tags is None:
            selected_top_tags = [tag for tag, _ in sorted_threshold_kept]
        else:
            selected_top_tags = [tag for tag, _ in sorted_threshold_kept[: self.top_k_tags]]

        coverage['top_k_selected_tag_count'] = len(selected_top_tags)
        coverage['top_k_selected_tags'] = selected_top_tags
        coverage['threshold_kept_tag_count'] = len(threshold_kept_counts)
        
        return coverage
    
    def get_missing_tags(self, builds: List[Dict]) -> List[str]:
        """
        Get blacklist tags that are not found in builds.
        
        Args:
            builds: List of builds to search
        
        Returns:
            List of tags not found in any build
        """
        found_tags = set()
        for build in builds:
            for tag in self._normalize_tags(build.get('tags', [])):
                if tag in self.blacklist_set:
                    found_tags.add(tag)
        
        missing = list(self.blacklist_set - found_tags)
        return sorted(missing)
    
    def preview_filtering(self, builds: List[Dict], n_samples: int = 3) -> Dict:
        """
        Preview blacklist filtering behavior before applying.
        
        Args:
            builds: List of builds to preview
            n_samples: Number of example builds to show
        
        Returns:
            Dictionary with preview information
        """
        preview = {
            'blacklist_tags': self.blacklist_tags,
            'min_tag_occurrences': self.min_tag_occurrences,
            'top_k_tags': self.top_k_tags,
            'total_builds': len(builds),
            'threshold_dropped_tag_count': 0,
            'threshold_dropped_tags': [],
            'top_k_selected_tag_count': 0,
            'top_k_selected_tags': [],
            'example_builds': []
        }

        pre_rows = []
        post_blacklist_counts: Counter[str] = Counter()

        for build in builds:
            original_tags = self._normalize_tags(build.get('tags', []))
            removed_by_blacklist = [tag for tag in original_tags if tag in self.blacklist_set]
            kept_after_blacklist = [tag for tag in original_tags if tag not in self.blacklist_set]

            post_blacklist_counts.update(kept_after_blacklist)
            pre_rows.append({
                'title': build.get('title', 'Unknown'),
                'original_tags': original_tags,
                'removed_by_blacklist': removed_by_blacklist,
                'kept_after_blacklist': kept_after_blacklist,
            })

        threshold_dropped_set = {
            tag
            for tag, count in post_blacklist_counts.items()
            if count < self.min_tag_occurrences
        }
        preview['threshold_dropped_tag_count'] = len(threshold_dropped_set)
        preview['threshold_dropped_tags'] = sorted(threshold_dropped_set)

        threshold_kept_counts: Counter[str] = Counter()
        for row in pre_rows:
            kept_after_threshold = [
                tag for tag in row['kept_after_blacklist'] if tag not in threshold_dropped_set
            ]
            row['kept_after_threshold'] = kept_after_threshold
            threshold_kept_counts.update(kept_after_threshold)

        sorted_threshold_kept = sorted(
            threshold_kept_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )
        if self.top_k_tags is None:
            selected_top_tags = [tag for tag, _ in sorted_threshold_kept]
        else:
            selected_top_tags = [tag for tag, _ in sorted_threshold_kept[: self.top_k_tags]]
        selected_top_set = set(selected_top_tags)

        preview['top_k_selected_tag_count'] = len(selected_top_tags)
        preview['top_k_selected_tags'] = selected_top_tags

        for row in pre_rows[:n_samples]:
            removed_by_threshold = [
                tag for tag in row['kept_after_blacklist'] if tag in threshold_dropped_set
            ]
            removed_by_top_k = [
                tag for tag in row['kept_after_threshold'] if tag not in selected_top_set
            ]
            kept_tags = [
                tag for tag in row['kept_after_threshold'] if tag in selected_top_set
            ]

            tags_to_remove = self._deduplicate_keep_order(
                row['removed_by_blacklist'] + removed_by_threshold + removed_by_top_k
            )

            preview['example_builds'].append({
                'title': row['title'],
                'original_tags': row['original_tags'],
                'tags_to_keep': kept_tags,
                'tags_to_remove': tags_to_remove,
                'removed_by_blacklist': row['removed_by_blacklist'],
                'removed_by_threshold': removed_by_threshold,
                'removed_by_top_k': removed_by_top_k,
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
                'filter_mode': 'blacklist',
                'blacklist_tags': self.blacklist_tags,
                'min_tag_occurrences': self.min_tag_occurrences,
                'top_k_tags': self.top_k_tags,
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

