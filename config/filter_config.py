"""Blacklist configuration used by tag filtering in dataset generation."""

# Tags listed here will be removed from each build during processing.
BLACKLIST_TAGS = [
    "medieval",
    "youtube",
    "medieval house"
]


def get_blacklist_tags() -> list[str]:
    """Return normalized blacklist tags with duplicates removed."""
    normalized: list[str] = []
    for raw in BLACKLIST_TAGS:
        tag = str(raw).strip()
        if not tag:
            continue
        normalized.append(tag)
    return list(dict.fromkeys(normalized))


if __name__ == "__main__":
    tags = get_blacklist_tags()
    print(f"Configured blacklist tags: {len(tags)}")
    for idx, tag in enumerate(tags, 1):
        print(f"  {idx:2d}. {tag}")
