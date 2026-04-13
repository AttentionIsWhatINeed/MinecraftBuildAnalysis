import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class DataProcessVisualizer:

    @staticmethod
    def _get_tags(build: Dict) -> List[str]:
        tags = build.get("tags", [])
        if not isinstance(tags, list):
            return []
        return [str(tag) for tag in tags if str(tag).strip()]

    @staticmethod
    def _get_image_count(build: Dict) -> int:
        images_count = build.get("images_count", 0)
        if isinstance(images_count, int):
            return max(images_count, 0)

        local_image_paths = build.get("local_image_paths", [])
        if isinstance(local_image_paths, list):
            return len(local_image_paths)

        return 0

    def _collect_stats(self, builds: List[Dict]) -> Dict:
        tag_counter: Counter[str] = Counter()
        tags_per_build: List[int] = []
        images_per_build: List[int] = []

        for build in builds:
            tags = self._get_tags(build)
            tag_counter.update(tags)
            tags_per_build.append(len(tags))
            images_per_build.append(self._get_image_count(build))

        return {
            "tag_counter": tag_counter,
            "tags_per_build": tags_per_build,
            "images_per_build": images_per_build,
        }

    def _collect_examples(self, builds: List[Dict], n: int) -> List[Dict]:
        examples: List[Dict] = []
        for build in builds[: max(n, 0)]:
            examples.append(
                {
                    "title": str(build.get("title", "")),
                    "build_url": str(build.get("build_url", "")),
                    "tags": self._get_tags(build),
                    "tag_count": len(self._get_tags(build)),
                    "image_count": self._get_image_count(build),
                }
            )
        return examples

    @staticmethod
    def _save_json(path: Path, payload: Dict | List) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _plot_tag_frequency(tag_rows: List[tuple], title: str, output_path: Path) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        if tag_rows:
            tag_names = [tag for tag, _ in tag_rows][::-1]
            tag_counts = [count for _, count in tag_rows][::-1]
            plt.barh(tag_names, tag_counts, color="#2A9D8F")
            plt.xlabel("Count")
            plt.ylabel("Tag")
            plt.grid(axis="x", linestyle="--", alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No tags", ha="center", va="center", transform=plt.gca().transAxes)
            plt.axis("off")

        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_distribution(values: List[int], title: str, xlabel: str, output_path: Path) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        if values:
            max_value = max(values)
            bins = range(0, max_value + 2)
            plt.hist(values, bins=bins, align="left", color="#3E7CB1", edgecolor="black", alpha=0.85)
            plt.xticks(range(0, max_value + 1))
            plt.xlabel(xlabel)
            plt.ylabel("Number of Builds")
            plt.grid(axis="y", linestyle="--", alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center", transform=plt.gca().transAxes)
            plt.axis("off")

        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    def save_before_after_visualizations(
        self,
        before_builds: List[Dict],
        after_builds: List[Dict],
        output_dir: str,
        before_top_n: int = 20,
        sample_n: int = 8,
    ) -> Dict[str, str]:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            ) from exc

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        before_stats = self._collect_stats(before_builds)
        after_stats = self._collect_stats(after_builds)

        before_tag_counter: Counter[str] = before_stats["tag_counter"]
        after_tag_counter: Counter[str] = after_stats["tag_counter"]

        before_top_tags = before_tag_counter.most_common(max(before_top_n, 0))
        after_all_tags = after_tag_counter.most_common()

        before_examples = self._collect_examples(before_builds, sample_n)
        after_examples = self._collect_examples(after_builds, sample_n)

        before_top_tags_json = out / "before_top20_tags_sorted.json"
        after_tags_json = out / "after_tags_sorted.json"
        before_examples_json = out / "before_example_8_builds_tags.json"
        after_examples_json = out / "after_example_8_builds_tags.json"

        self._save_json(before_top_tags_json, [{"tag": t, "count": c} for t, c in before_top_tags])
        self._save_json(after_tags_json, [{"tag": t, "count": c} for t, c in after_all_tags])
        self._save_json(before_examples_json, before_examples)
        self._save_json(after_examples_json, after_examples)

        before_top_tags_plot = out / "before_top20_tags_barh.png"
        before_tags_dist_plot = out / "before_tags_per_build_hist.png"
        before_images_dist_plot = out / "before_images_per_build_hist.png"

        after_tags_plot = out / "after_tags_frequency_barh.png"
        after_tags_dist_plot = out / "after_tags_per_build_hist.png"
        after_images_dist_plot = out / "after_images_per_build_hist.png"

        self._plot_tag_frequency(
            before_top_tags,
            title="Before Processing: Top 20 Tag Frequencies",
            output_path=before_top_tags_plot,
        )
        self._plot_distribution(
            before_stats["tags_per_build"],
            title="Before Processing: Tag Count Distribution per Build",
            xlabel="Tags per Build",
            output_path=before_tags_dist_plot,
        )
        self._plot_distribution(
            before_stats["images_per_build"],
            title="Before Processing: Image Count Distribution per Build",
            xlabel="Images per Build",
            output_path=before_images_dist_plot,
        )

        self._plot_tag_frequency(
            after_all_tags,
            title="After Processing: Tag Frequency (Sorted)",
            output_path=after_tags_plot,
        )
        self._plot_distribution(
            after_stats["tags_per_build"],
            title="After Processing: Tag Count Distribution per Build",
            xlabel="Tags per Build",
            output_path=after_tags_dist_plot,
        )
        self._plot_distribution(
            after_stats["images_per_build"],
            title="After Processing: Image Count Distribution per Build",
            xlabel="Images per Build",
            output_path=after_images_dist_plot,
        )

        summary_json = out / "before_after_visualization_summary.json"
        self._save_json(
            summary_json,
            {
                "generated_at": datetime.now().isoformat(),
                "before": {
                    "num_builds": len(before_builds),
                    "unique_tags": len(before_tag_counter),
                    "top20_tags": [{"tag": t, "count": c} for t, c in before_top_tags],
                },
                "after": {
                    "num_builds": len(after_builds),
                    "unique_tags": len(after_tag_counter),
                    "tags_sorted": [{"tag": t, "count": c} for t, c in after_all_tags],
                },
            },
        )

        return {
            "before_top20_tags_sorted_json": str(before_top_tags_json),
            "before_top20_tags_barh": str(before_top_tags_plot),
            "before_example_8_builds_tags_json": str(before_examples_json),
            "before_tags_per_build_hist": str(before_tags_dist_plot),
            "before_images_per_build_hist": str(before_images_dist_plot),
            "after_tags_sorted_json": str(after_tags_json),
            "after_tags_frequency_barh": str(after_tags_plot),
            "after_example_8_builds_tags_json": str(after_examples_json),
            "after_tags_per_build_hist": str(after_tags_dist_plot),
            "after_images_per_build_hist": str(after_images_dist_plot),
            "summary_json": str(summary_json),
        }
