import json
from pathlib import Path

from src.process.data_processor import DataProcessor


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_filter_valid_builds_deduplicates_and_keeps_higher_quality(tmp_path: Path) -> None:
    metadata_file = tmp_path / "builds_metadata.json"

    low_quality_duplicate = {
        "title": "low_quality_duplicate",
        "build_url": "https://example.com/build/1",
        "tags": ["a"],
        "local_image_paths": ["img_a.jpg"],
        "images_count": 1,
    }
    high_quality_duplicate = {
        "title": "high_quality_duplicate",
        "build_url": "https://example.com/build/1",
        "tags": ["a", "b"],
        "local_image_paths": ["img_a.jpg", "img_b.jpg"],
        "images_count": 3,
    }
    no_tags = {
        "title": "no_tags",
        "build_url": "https://example.com/build/2",
        "tags": [],
        "local_image_paths": ["img_c.jpg"],
        "images_count": 1,
    }
    no_images = {
        "title": "no_images",
        "build_url": "https://example.com/build/3",
        "tags": ["c"],
        "local_image_paths": [],
        "images_count": 0,
    }

    _write_json(
        metadata_file,
        [low_quality_duplicate, high_quality_duplicate, no_tags, no_images],
    )

    processor = DataProcessor(metadata_file=str(metadata_file))
    filtered = processor.filter_valid_builds(
        require_tags=True,
        require_images=True,
        require_existing_files=False,
    )

    assert len(filtered) == 1
    assert filtered[0]["title"] == "high_quality_duplicate"

    report = processor.last_filter_report
    assert report["raw_total"] == 4
    assert report["valid_before_dedup"] == 2
    assert report["duplicates_removed"] == 1
    assert report["valid_after_dedup"] == 1


def test_get_statistics_computes_expected_values(tmp_path: Path) -> None:
    metadata_file = tmp_path / "builds_metadata.json"

    payload = [
        {
            "title": "build_a",
            "build_url": "https://example.com/a",
            "tags": ["x", "y"],
            "local_image_paths": ["a1.jpg", "a2.jpg"],
            "images_count": 2,
        },
        {
            "title": "build_b",
            "build_url": "https://example.com/b",
            "tags": ["y"],
            "local_image_paths": ["b1.jpg"],
            "images_count": 1,
        },
    ]
    _write_json(metadata_file, payload)

    processor = DataProcessor(metadata_file=str(metadata_file))
    stats = processor.get_statistics(payload)

    assert stats["total_builds"] == 2
    assert stats["total_images"] == 3
    assert stats["unique_tags"] == 2
    assert stats["total_tag_occurrences"] == 3
    assert stats["avg_images_per_build"] == 1.5
    assert stats["avg_tags_per_build"] == 1.5
