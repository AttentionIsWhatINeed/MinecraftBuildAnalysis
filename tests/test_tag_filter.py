from src.process.tag_filter import TagFilter


def _build(title: str, tags: list[str]) -> dict:
    return {
        "title": title,
        "build_url": f"https://example.com/{title}",
        "tags": tags,
        "local_image_paths": [f"{title}.jpg"],
        "images_count": 1,
    }


def test_filter_builds_applies_blacklist_threshold_and_top_k() -> None:
    builds = [
        _build("b1", ["blocked", "keep_a", "rare"]),
        _build("b2", ["keep_a", "keep_b"]),
        _build("b3", ["keep_a", "keep_b"]),
        _build("b4", ["blocked", "keep_b"]),
    ]

    tag_filter = TagFilter(
        blacklist_tags=["blocked"],
        split_tags=False,
        min_tag_occurrences=2,
        top_k_tags=1,
    )

    filtered = tag_filter.filter_builds_by_tags(builds, remove_tag_mismatch=True)

    assert len(filtered) == 3
    assert all(b["tags"] == ["keep_a"] for b in filtered)

    report = tag_filter.last_filter_report
    assert report["min_tag_occurrences"] == 2
    assert report["top_k_requested"] == 1
    assert report["selected_top_tags"] == ["keep_a"]


def test_create_train_val_test_split_is_reproducible() -> None:
    builds = [_build(f"build_{i}", ["keep"]) for i in range(10)]

    tag_filter = TagFilter(blacklist_tags=[])

    train_a, val_a, test_a = tag_filter.create_train_val_test_split(
        builds,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=13,
    )
    train_b, val_b, test_b = tag_filter.create_train_val_test_split(
        builds,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=13,
    )

    assert len(train_a) == 7
    assert len(val_a) == 2
    assert len(test_a) == 1

    assert [b["title"] for b in train_a] == [b["title"] for b in train_b]
    assert [b["title"] for b in val_a] == [b["title"] for b in val_b]
    assert [b["title"] for b in test_a] == [b["title"] for b in test_b]
