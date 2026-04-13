import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processor for cleaning and filtering Minecraft build metadata.
    
    This class handles:
    - Loading raw metadata
    - Filtering datasets (removing builds without tags or images)
    - Deduplicating builds (prefers unique build_url)
    - Validating image file existence
    - Generating statistics and tag distribution
    - Saving processed datasets
    """
    
    def __init__(
        self,
        metadata_file: str = "data/raw/metadata/builds_metadata.json"
    ):
        """
        Initialize DataProcessor.
        
        Args:
            metadata_file: Path to the raw metadata JSON file
        """
        self.metadata_file = Path(metadata_file)
        self.raw_data: List[Dict] = []
        self.processed_data: List[Dict] = []
        self.last_filter_report: Dict[str, int] = {}
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        self._load_metadata()
        logger.info(f"Loaded {len(self.raw_data)} builds from {self.metadata_file}")

    def _normalize_key_value(self, value: str) -> str:
        """Normalize key text for deduplication comparisons."""
        return value.strip().lower().replace('\\', '/')

    def _build_identity_key(self, build: Dict) -> Optional[str]:
        """Create a stable identity key for deduplicating builds."""
        build_url = self._normalize_key_value(str(build.get('build_url', '')))
        if build_url:
            return f"url::{build_url}"

        build_dir = self._normalize_key_value(str(build.get('build_directory', '')))
        if build_dir:
            return f"dir::{build_dir}"

        local_image_paths = build.get('local_image_paths', []) or []
        if local_image_paths:
            first_image = self._normalize_key_value(str(local_image_paths[0]))
            if first_image:
                return f"img::{first_image}"

        title = self._normalize_key_value(str(build.get('title', '')))
        if title:
            return f"title::{title}"

        return None

    def _build_quality_score(self, build: Dict) -> tuple[int, int, int]:
        """Score a build record quality; higher score is preferred when deduplicating."""
        local_image_paths = build.get('local_image_paths', []) or []
        tags = build.get('tags', []) or []

        images_count = build.get('images_count', 0)
        if not isinstance(images_count, int):
            try:
                images_count = int(images_count)
            except (TypeError, ValueError):
                images_count = 0

        return (images_count, len(local_image_paths), len(tags))

    def _deduplicate_builds(self, builds: List[Dict]) -> tuple[List[Dict], int]:
        """Remove duplicate builds while preserving order of first-seen identities."""
        if not builds:
            return builds, 0

        deduped: List[Dict] = []
        key_to_index: Dict[str, int] = {}
        key_to_quality: Dict[str, tuple[int, int, int]] = {}
        duplicates_removed = 0

        for build in builds:
            key = self._build_identity_key(build)

            # Keep records without reliable identity key; only keyed records are deduplicated.
            if key is None:
                deduped.append(build)
                continue

            quality = self._build_quality_score(build)
            existing_idx = key_to_index.get(key)

            if existing_idx is None:
                key_to_index[key] = len(deduped)
                key_to_quality[key] = quality
                deduped.append(build)
                continue

            duplicates_removed += 1
            if quality > key_to_quality[key]:
                deduped[existing_idx] = build
                key_to_quality[key] = quality

        return deduped, duplicates_removed
    
    def _load_metadata(self):
        """Load metadata from JSON file."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            logger.info(f"Successfully loaded metadata with {len(self.raw_data)} entries")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def filter_valid_builds(
        self,
        require_tags: bool = True,
        require_images: bool = True,
        require_existing_files: bool = True,
        min_tags: int = 1,
        min_images: int = 1
    ) -> List[Dict]:
        """
        Filter and validate builds based on criteria.
        
        Args:
            require_tags: If True, exclude builds without tags
            require_images: If True, exclude builds without image paths
            require_existing_files: If True, verify image files physically exist
            min_tags: Minimum number of tags required per build
            min_images: Minimum number of images required per build
        
        Returns:
            List of valid builds meeting all criteria
        """
        logger.info("Starting data filtering...")
        valid_builds = []
        
        for idx, build in enumerate(self.raw_data):
            is_valid = True
            reasons = []
            
            # Check tags
            tags = build.get('tags', [])
            if require_tags and (not tags or len(tags) < min_tags):
                is_valid = False
                reasons.append(f"tags ({len(tags)} < {min_tags})")
            
            # Check images
            image_paths = build.get('local_image_paths', [])
            if require_images and (not image_paths or len(image_paths) < min_images):
                is_valid = False
                reasons.append(f"images ({len(image_paths)} < {min_images})")
            
            # Check if image files exist
            if is_valid and require_existing_files:
                for image_path in image_paths:
                    if not Path(image_path).exists():
                        is_valid = False
                        reasons.append(f"missing file: {image_path}")
                        break
            
            if is_valid:
                valid_builds.append(build)
            else:
                if idx < 10 or idx % 100 == 0:  # Log first 10 and every 100th
                    logger.debug(f"Build {idx} ({build.get('title', 'Unknown')}): "
                                f"Filtered out - {', '.join(reasons)}")
        
        valid_before_dedup = len(valid_builds)
        valid_builds, duplicates_removed = self._deduplicate_builds(valid_builds)
        if duplicates_removed > 0:
            logger.info(f"Deduplication removed {duplicates_removed} duplicate builds")

        self.processed_data = valid_builds
        
        total = len(self.raw_data)
        valid = len(valid_builds)
        retention_pct = (100 * valid / total) if total else 0.0
        logger.info(f"Filtering complete: {valid}/{total} builds valid "
                   f"({retention_pct:.1f}% retention)")

        self.last_filter_report = {
            'raw_total': total,
            'valid_before_dedup': valid_before_dedup,
            'duplicates_removed': duplicates_removed,
            'valid_after_dedup': valid,
        }
        
        return valid_builds
    
    def get_statistics(self, data: Optional[List[Dict]] = None) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            data: List of builds to analyze. If None, uses processed_data
        
        Returns:
            Dictionary with statistics
        """
        if data is None:
            data = self.processed_data if self.processed_data else self.raw_data
        
        if not data:
            return {}
        
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
            'min_images': min(image_counts),
            'max_images': max(image_counts),
        }
        
        return stats
    
    def get_split_tags_distribution(self, data: Optional[List[Dict]] = None, n: int = 50) -> List[tuple]:
        """
        Get distribution of split tags (tags split by space into individual words).
        
        This analyzes what tags would be created if multi-word tags were split.
        Useful for deciding which split tags to use in filtering.
        
        Args:
            data: List of builds to analyze. If None, uses processed_data
            n: Number of top split tags to return
        
        Returns:
            List of (tag, count) tuples sorted by count descending
        """
        if data is None:
            data = self.processed_data if self.processed_data else self.raw_data
        
        split_tag_counts = {}
        
        for build in data:
            for tag in build.get('tags', []):
                # Split each tag by space
                words = tag.split()
                for word in words:
                    split_tag_counts[word] = split_tag_counts.get(word, 0) + 1
        
        sorted_tags = sorted(split_tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:n]
    
    def get_top_tags(self, n: int = 20, data: Optional[List[Dict]] = None) -> List[tuple]:
        """
        Get the top N most common tags.
        
        Args:
            n: Number of top tags to return
            data: List of builds to analyze. If None, uses processed_data
        
        Returns:
            List of (tag, count) tuples sorted by count descending
        """
        if data is None:
            data = self.processed_data if self.processed_data else self.raw_data
        
        tag_counts = {}
        for build in data:
            for tag in build.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:n]
    
    def save_processed_dataset(
        self,
        output_file: str,
        data: Optional[List[Dict]] = None,
        include_stats: bool = True
    ) -> Path:
        """
        Save processed dataset to JSON file.
        
        Args:
            output_file: Path where to save the dataset
            data: List of builds to save. If None, uses processed_data
            include_stats: If True, include statistics in the output
        
        Returns:
            Path to the saved file
        """
        if data is None:
            data = self.processed_data
        
        if not data:
            logger.warning("No data to save!")
            return Path(output_file)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_builds': len(data),
                'total_images': sum(b.get('images_count', 0) for b in data),
            },
            'builds': data
        }
        
        if include_stats:
            output['statistics'] = self.get_statistics(data)
            output['top_tags'] = [
                {'tag': tag, 'count': count}
                for tag, count in self.get_top_tags(n=50, data=data)
            ]
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved processed dataset with {len(data)} builds to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def export_summary_report(self, output_file: str, data: Optional[List[Dict]] = None):
        """
        Export a summary report about the dataset.
        
        Args:
            output_file: Path where to save the report
            data: List of builds to analyze. If None, uses processed_data
        """
        if data is None:
            data = self.processed_data
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_statistics(data)
        top_tags = self.get_top_tags(n=20, data=data)
        
        report = []
        report.append("=" * 60)
        report.append("MINECRAFT BUILD ANALYSIS - DATA REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("DATASET OVERVIEW")
        report.append("-" * 60)
        report.append(f"Total Builds: {stats.get('total_builds', 0)}")
        report.append(f"Total Images: {stats.get('total_images', 0)}")
        report.append(f"Average Images per Build: {stats.get('avg_images_per_build', 0):.2f}")
        report.append(f"Image Range: {stats.get('min_images', 0)} - {stats.get('max_images', 0)}")
        report.append("")
        
        report.append("TAG STATISTICS")
        report.append("-" * 60)
        report.append(f"Unique Tags: {stats.get('unique_tags', 0)}")
        report.append(f"Total Tag Occurrences: {stats.get('total_tag_occurrences', 0)}")
        report.append(f"Average Tags per Build: {stats.get('avg_tags_per_build', 0):.2f}")
        report.append("")
        
        report.append("TOP 20 TAGS")
        report.append("-" * 60)
        for i, (tag, count) in enumerate(top_tags, 1):
            report.append(f"{i:2d}. {tag:30s} : {count:5d} occurrences")
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Summary report saved to {output_path}")
        print(report_text)  # Also print to console