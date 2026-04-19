import json
from pathlib import Path

from src.process.data_processor import DataProcessor


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_processor(tmp_path: Path, payload) -> DataProcessor:
    metadata_file = tmp_path / "builds_metadata.json"
    _write_json(metadata_file, payload)
    return DataProcessor(metadata_file=str(metadata_file))


def test_identity_key_quality_and_dedup_helpers(tmp_path: Path) -> None:
    processor = _make_processor(tmp_path, [])

    assert processor._build_identity_key({"build_url": " HTTPS://EXAMPLE.COM/A\\B "}) == "url::https://example.com/a/b"
    assert processor._build_identity_key({"build_directory": " SOME\\DIR "}) == "dir::some/dir"
    assert processor._build_identity_key({"local_image_paths": [" IMG\\ONE.PNG "]}) == "img::img/one.png"
    assert processor._build_identity_key({"title": " Demo "}) == "title::demo"
    assert processor._build_identity_key({}) is None

    assert processor._build_quality_score({"images_count": "3", "local_image_paths": ["a", "b"], "tags": ["x"]}) == (3, 2, 1)
    assert processor._build_quality_score({"images_count": "bad", "local_image_paths": [], "tags": []}) == (0, 0, 0)

    deduped, removed = processor._deduplicate_builds(
        [
            {"title": "first", "build_url": "https://x/a", "images_count": 1, "local_image_paths": ["a"], "tags": ["t"]},
            {"title": "better", "build_url": "https://x/a", "images_count": 2, "local_image_paths": ["a", "b"], "tags": ["t", "u"]},
            {"title": "no_key_1"},
            {"title": "no_key_2"},
        ]
    )
    assert removed == 1
    assert deduped[0]["title"] == "better"
    assert [d["title"] for d in deduped[1:]] == ["no_key_1", "no_key_2"]


def test_filter_valid_builds_with_existing_file_checks(tmp_path: Path) -> None:
    existing = tmp_path / "exists.jpg"
    existing.write_text("ok", encoding="utf-8")

    payload = [
        {
            "title": "valid",
            "build_url": "https://example.com/valid",
            "tags": ["a", "b"],
            "local_image_paths": [str(existing)],
            "images_count": 1,
        },
        {
            "title": "missing_file",
            "build_url": "https://example.com/missing",
            "tags": ["a", "b"],
            "local_image_paths": [str(tmp_path / "missing.jpg")],
            "images_count": 1,
        },
        {
            "title": "few_tags",
            "build_url": "https://example.com/few_tags",
            "tags": ["a"],
            "local_image_paths": [str(existing)],
            "images_count": 1,
        },
    ]
    processor = _make_processor(tmp_path, payload)

    filtered = processor.filter_valid_builds(
        require_existing_files=True,
        min_tags=2,
        min_images=1,
    )

    assert [b["title"] for b in filtered] == ["valid"]
    assert processor.processed_data == filtered
    assert processor.last_filter_report["raw_total"] == 3
    assert processor.last_filter_report["valid_after_dedup"] == 1


def test_statistics_and_tag_distribution_apis(tmp_path: Path) -> None:
    payload = [
        {"title": "a", "build_url": "https://x/a", "tags": ["red stone", "castle"], "local_image_paths": ["a.jpg"], "images_count": 1},
        {"title": "b", "build_url": "https://x/b", "tags": ["red stone"], "local_image_paths": ["b.jpg", "c.jpg"], "images_count": 2},
    ]
    processor = _make_processor(tmp_path, payload)
    processor.processed_data = payload

    assert processor.get_statistics([]) == {}
    assert processor.get_top_tags(n=1) == [("red stone", 2)]
    assert processor.get_split_tags_distribution(n=2) == [("red", 2), ("stone", 2)]


def test_save_and_export_reports(tmp_path: Path) -> None:
    payload = [
        {"title": "a", "build_url": "https://x/a", "tags": ["x"], "local_image_paths": ["a.jpg"], "images_count": 1},
        {"title": "b", "build_url": "https://x/b", "tags": ["y", "x"], "local_image_paths": ["b.jpg"], "images_count": 1},
    ]
    processor = _make_processor(tmp_path, payload)
    processor.processed_data = payload

    empty_out = tmp_path / "out" / "empty.json"
    returned = processor.save_processed_dataset(str(empty_out), data=[])
    assert returned == empty_out
    assert not empty_out.exists()

    out_no_stats = tmp_path / "out" / "processed_no_stats.json"
    processor.save_processed_dataset(str(out_no_stats), data=payload, include_stats=False)
    parsed_no_stats = json.loads(out_no_stats.read_text(encoding="utf-8"))
    assert parsed_no_stats["metadata"]["total_builds"] == 2
    assert "statistics" not in parsed_no_stats
    assert "top_tags" not in parsed_no_stats

    out_with_stats = tmp_path / "out" / "processed_with_stats.json"
    processor.save_processed_dataset(str(out_with_stats), data=payload, include_stats=True)
    parsed_with_stats = json.loads(out_with_stats.read_text(encoding="utf-8"))
    assert parsed_with_stats["statistics"]["unique_tags"] == 2
    assert parsed_with_stats["top_tags"][0]["tag"] == "x"
    assert parsed_with_stats["top_tags"][0]["count"] == 2

    report_file = tmp_path / "out" / "summary.txt"
    processor.export_summary_report(str(report_file), data=payload)
    report_text = report_file.read_text(encoding="utf-8")
    assert "DATASET OVERVIEW" in report_text
    assert "TOP 20 TAGS" in report_text
