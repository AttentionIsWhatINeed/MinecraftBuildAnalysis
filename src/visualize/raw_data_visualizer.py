import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class RawVizSummary:
    total_builds: int
    total_images: int
    unique_tags: int
    avg_images_per_build: float
    avg_tags_per_build: float
    min_images_per_build: int
    max_images_per_build: int


class RawDataVisualizer:
    """Create exploratory visualizations for raw Minecraft build metadata."""

    def __init__(self, metadata_file: str = "data/raw/metadata/builds_metadata.json"):
        self.metadata_file = Path(metadata_file)
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        if not isinstance(self.raw_data, list):
            raise ValueError("Raw metadata must be a list of build objects.")

    @staticmethod
    def _get_build_tags(build: Dict) -> List[str]:
        tags = build.get("tags", [])
        if not isinstance(tags, list):
            return []
        return [str(t) for t in tags]

    @staticmethod
    def _get_image_count(build: Dict) -> int:
        if "images_count" in build and isinstance(build["images_count"], int):
            return build["images_count"]
        image_paths = build.get("local_image_paths", [])
        if isinstance(image_paths, list):
            return len(image_paths)
        return 0

    def _collect_core_stats(self) -> Dict:
        image_counts = [self._get_image_count(b) for b in self.raw_data]
        tag_lists = [self._get_build_tags(b) for b in self.raw_data]

        all_tags: List[str] = []
        for tags in tag_lists:
            all_tags.extend(tags)

        tags_per_build = [len(tags) for tags in tag_lists]
        tag_counter = Counter(all_tags)

        category_counter: Counter[str] = Counter()
        for build in self.raw_data:
            category_url = build.get("category_url", "")
            if isinstance(category_url, str) and category_url.strip():
                category_name = category_url.rstrip("/").split("/")[-1].strip()
                if category_name:
                    category_counter[category_name] += 1

        return {
            "image_counts": image_counts,
            "tags_per_build": tags_per_build,
            "tag_counter": tag_counter,
            "category_counter": category_counter,
        }

    def get_summary(self) -> RawVizSummary:
        stats = self._collect_core_stats()
        image_counts = stats["image_counts"]
        tag_counter: Counter = stats["tag_counter"]
        tags_per_build = stats["tags_per_build"]

        total_builds = len(self.raw_data)
        total_images = sum(image_counts)

        return RawVizSummary(
            total_builds=total_builds,
            total_images=total_images,
            unique_tags=len(tag_counter),
            avg_images_per_build=(total_images / total_builds) if total_builds else 0.0,
            avg_tags_per_build=(sum(tags_per_build) / total_builds) if total_builds else 0.0,
            min_images_per_build=min(image_counts) if image_counts else 0,
            max_images_per_build=max(image_counts) if image_counts else 0,
        )

    def save_visualizations(self, output_dir: str = "outputs/raw_visualization", top_n_tags: int = 20) -> Dict[str, str]:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            ) from exc

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stats = self._collect_core_stats()
        image_counts = stats["image_counts"]
        tags_per_build = stats["tags_per_build"]
        tag_counter: Counter = stats["tag_counter"]
        category_counter: Counter = stats["category_counter"]

        files: Dict[str, str] = {}

        # 1) Distribution of images per build
        plt.figure(figsize=(10, 6))
        plt.hist(image_counts, bins=20, color="#3E7CB1", edgecolor="black", alpha=0.85)
        plt.title("Raw Data: Images per Build Distribution")
        plt.xlabel("Images per Build")
        plt.ylabel("Number of Builds")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        path_images = out / "raw_images_per_build_hist.png"
        plt.tight_layout()
        plt.savefig(path_images, dpi=200)
        plt.close()
        files["images_per_build_hist"] = str(path_images)

        # 2) Top tag frequencies
        top_tags = tag_counter.most_common(top_n_tags)
        top_tag_names = [t for t, _ in top_tags][::-1]
        top_tag_counts = [c for _, c in top_tags][::-1]

        plt.figure(figsize=(12, 8))
        plt.barh(top_tag_names, top_tag_counts, color="#2A9D8F")
        plt.title(f"Raw Data: Top {top_n_tags} Tag Frequencies")
        plt.xlabel("Count")
        plt.ylabel("Tag")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        path_tags = out / "raw_top_tags_barh.png"
        plt.tight_layout()
        plt.savefig(path_tags, dpi=200)
        plt.close()
        files["top_tags_barh"] = str(path_tags)

        # 3) Distribution of tags per build
        plt.figure(figsize=(10, 6))
        max_tags = max(tags_per_build) if tags_per_build else 0
        bins = range(0, max_tags + 2)
        plt.hist(tags_per_build, bins=bins, align="left", color="#E76F51", edgecolor="black", alpha=0.85)
        plt.title("Raw Data: Tags per Build Distribution")
        plt.xlabel("Tags per Build")
        plt.ylabel("Number of Builds")
        plt.xticks(range(0, max_tags + 1))
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        path_tagdist = out / "raw_tags_per_build_hist.png"
        plt.tight_layout()
        plt.savefig(path_tagdist, dpi=200)
        plt.close()
        files["tags_per_build_hist"] = str(path_tagdist)

        # 4) Top categories by build count
        top_categories = category_counter.most_common(15)
        cat_names = [n for n, _ in top_categories][::-1]
        cat_counts = [c for _, c in top_categories][::-1]

        plt.figure(figsize=(12, 8))
        plt.barh(cat_names, cat_counts, color="#F4A261")
        plt.title("Raw Data: Top 15 Categories by Build Count")
        plt.xlabel("Build Count")
        plt.ylabel("Category")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        path_category = out / "raw_top_categories_barh.png"
        plt.tight_layout()
        plt.savefig(path_category, dpi=200)
        plt.close()
        files["top_categories_barh"] = str(path_category)

        # 5) Save summary JSON
        summary = self.get_summary()
        summary_path = out / "raw_data_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "metadata_file": str(self.metadata_file),
                    "summary": {
                        "total_builds": summary.total_builds,
                        "total_images": summary.total_images,
                        "unique_tags": summary.unique_tags,
                        "avg_images_per_build": summary.avg_images_per_build,
                        "avg_tags_per_build": summary.avg_tags_per_build,
                        "min_images_per_build": summary.min_images_per_build,
                        "max_images_per_build": summary.max_images_per_build,
                    },
                    "top_tags": [{"tag": t, "count": c} for t, c in tag_counter.most_common(top_n_tags)],
                    "top_categories": [{"category": n, "count": c} for n, c in category_counter.most_common(15)],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        files["summary_json"] = str(summary_path)

        return files
