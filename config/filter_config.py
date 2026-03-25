"""
Filter configuration for dataset generation.

Edit the TAG_PRESETS below to customize which tags are used for filtering.
Then select a preset in generate_filtered_dataset.py or use these directly.
"""

# Predefined tag configurations
TAG_PRESETS = {
    "custom": [
        "medieval",
        "statue",
        "house",
        "fountain",
        "fantasy",
        "truck",
        "siege",
        "bridge",
        "vehicle",
        "obelisk"
    ],
    
    # Split tags presets (for use with enable_tag_splitting=True)
    # Generated from split tags analysis
    
    "split_top_10": [
        "medieval", "decoration", "statue", "house", "small",
        "fountain", "comics", "fantasy", "outdoor", "working"
    ],
    
    "split_top_20": [
        "medieval", "decoration", "statue", "house", "small",
        "fountain", "comics", "fantasy", "outdoor", "working",
        "truck", "unfurnished", "mechanism", "siege", "bridge",
        "vehicle", "obelisk", "steampunk", "building", "trailer"
    ],
    
    "split_top_30": [
        "medieval", "decoration", "statue", "house", "small",
        "fountain", "comics", "fantasy", "outdoor", "working",
        "truck", "unfurnished", "mechanism", "siege", "bridge",
        "vehicle", "obelisk", "steampunk", "building", "trailer",
        "church", "modern", "farm", "piston", "bus",
        "castle", "ruins", "character", "pixel", "art"
    ]
}


def get_tags(preset_name: str = "custom") -> list:
    """
    Get tags for a specific preset.
    
    Args:
        preset_name: Name of the preset to use
        
    Returns:
        List of tags
    """
    if preset_name not in TAG_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    tags = TAG_PRESETS[preset_name]
    
    if not tags:
        raise ValueError(f"Preset '{preset_name}' has no tags configured")
    
    return tags


def list_presets():
    """Print all available presets."""
    print("Available tag presets:")
    for name, tags in TAG_PRESETS.items():
        print(f"  '{name}': {len(tags)} tags")
        if tags:
            print(f"    {tags[:3]}{'...' if len(tags) > 3 else ''}")


if __name__ == "__main__":
    list_presets()
