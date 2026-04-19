import json
from pathlib import Path

import pytest

from src.process.tag_filter import TagFilter


def _build(title: str, tags: list[str]) -> dict:
    return {
        "title": title,
        "build_url": f"https://example.com/{title}",
        "tags": tags,
        "local_image_paths": [f"{title}.jpg"],
        "images_count": 1,
    }


def test_constructor_validation_and_split_blacklist_behavior() -> None:
    with pytest.raises(ValueError):
        TagFilter(blacklist_tags=[], min_tag_occurrences=0)

    with pytest.raises(ValueError):
        TagFilter(blacklist_tags=[], top_k_tags=0)

    tf = TagFilter(blacklist_tags=[" red stone ", "stone", "stone"], split_tags=True)
    assert tf.blacklist_tags == ["red", "stone"]
    assert tf.blacklist_set == {"red", "stone"}


def test_filter_with_remove_tag_mismatch_false_keeps_empty_tag_builds() -> None:
    builds = [
        _build("b1", ["blocked", "keep"]),
        _build("b2", ["blocked"]),
        _build("b3", ["keep", "rare"]),
    ]
    tf = TagFilter(
        blacklist_tags=["blocked"],
        split_tags=False,
        min_tag_occurrences=2,
        top_k_tags=1,
    )

    filtered = tf.filter_builds_by_tags(builds, remove_tag_mismatch=False)

    assert len(filtered) == 3
    assert [b["tags"] for b in filtered] == [["keep"], [], ["keep"]]
    assert tf.last_filter_report["removed_builds_by_blacklist"] == 0
    assert tf.last_filter_report["removed_builds_by_threshold"] == 0
    assert tf.last_filter_report["removed_builds_by_top_k"] == 0
    assert tf.last_filter_report["selected_top_tags"] == ["keep"]


def test_coverage_missing_preview_and_split_edge_cases() -> None:
    builds = [
        _build("a", ["blocked", "keep", "rare"]),
        _build("b", ["keep", "other"]),
        _build("c", ["keep"]),
    ]
    tf = TagFilter(
        blacklist_tags=["blocked", "never_seen"],
        min_tag_occurrences=2,
        top_k_tags=1,
    )

    coverage = tf.get_tag_coverage(builds)
    assert coverage["blacklist_size"] == 2
    assert coverage["blacklisted_tags_found_in_builds"] == 1
    assert coverage["coverage_percentage"] == 50.0
    assert coverage["low_frequency_tag_counts"] == {"other": 1, "rare": 1}
    assert coverage["threshold_kept_tag_count"] == 1
    assert coverage["top_k_selected_tags"] == ["keep"]

    assert tf.get_missing_tags(builds) == ["never_seen"]

    preview = tf.preview_filtering(builds, n_samples=2)
    assert preview["threshold_dropped_tags"] == ["other", "rare"]
    assert preview["top_k_selected_tags"] == ["keep"]
    assert len(preview["example_builds"]) == 2
    assert preview["example_builds"][0]["title"] == "a"
    assert "blocked" in preview["example_builds"][0]["tags_to_remove"]
    assert preview["example_builds"][1]["will_be_kept"] is True

    with pytest.raises(ValueError):
        tf.create_train_val_test_split(builds, train_ratio=0.6, val_ratio=0.3, test_ratio=0.2)

    empty_tf = TagFilter(blacklist_tags=["keep"])
    train, val, test = empty_tf.create_train_val_test_split(
        [_build("only", ["keep"])],
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
    )
    assert train == []
    assert val == []
    assert test == []


def test_save_processed_dataset_outputs_expected_structure(tmp_path: Path) -> None:
    tf = TagFilter(blacklist_tags=["blocked"], min_tag_occurrences=1, top_k_tags=None)
    builds = [
        _build("a", ["x", "y"]),
        _build("b", ["x"]),
    ]

    empty_out = tmp_path / "out" / "empty.json"
    returned = tf.save_processed_dataset(str(empty_out), data=[])
    assert returned == empty_out
    assert not empty_out.exists()

    output = tmp_path / "out" / "filtered.json"
    tf.save_processed_dataset(str(output), data=builds, include_stats=True)
    parsed = json.loads(output.read_text(encoding="utf-8"))

    assert parsed["metadata"]["filter_mode"] == "blacklist"
    assert parsed["metadata"]["blacklist_tags"] == ["blocked"]
    assert parsed["statistics"]["total_builds"] == 2
    assert parsed["statistics"]["unique_tags"] == 2
    assert parsed["tag_distribution"][0] == {"tag": "x", "count": 2}
