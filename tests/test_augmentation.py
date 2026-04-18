import pytest

from src.train.augmentation import (
    AugmentationConfig,
    build_eval_transform,
    build_train_eval_transforms,
    build_train_transform,
)


def _transform_type_names(compose_obj) -> list[str]:
    return [type(step).__name__ for step in compose_obj.transforms]


def test_default_train_augmentation_matches_current_behavior() -> None:
    train_tf = build_train_transform(image_size=224, config=AugmentationConfig())

    assert _transform_type_names(train_tf) == [
        "Resize",
        "RandomHorizontalFlip",
        "ColorJitter",
        "ToTensor",
        "Normalize",
    ]


def test_disable_augmentation_falls_back_to_eval_transform() -> None:
    train_tf = build_train_transform(
        image_size=224,
        config=AugmentationConfig(
            enabled=False,
            hflip_prob=1.0,
            color_jitter_strength=0.5,
            rotation_degrees=15.0,
            random_resized_crop_scale_min=0.8,
            random_erasing_prob=0.3,
        ),
    )
    eval_tf = build_eval_transform(image_size=224)

    assert _transform_type_names(train_tf) == _transform_type_names(eval_tf)


def test_stronger_augmentation_adds_rotation_crop_and_erasing() -> None:
    train_tf = build_train_transform(
        image_size=224,
        config=AugmentationConfig(
            enabled=True,
            hflip_prob=0.7,
            color_jitter_strength=0.1,
            rotation_degrees=12.0,
            random_resized_crop_scale_min=0.8,
            random_erasing_prob=0.2,
        ),
    )

    assert _transform_type_names(train_tf) == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "ToTensor",
        "Normalize",
        "RandomErasing",
    ]


def test_invalid_augmentation_arguments_raise_value_error() -> None:
    with pytest.raises(ValueError):
        build_train_transform(image_size=224, config=AugmentationConfig(hflip_prob=1.1))

    with pytest.raises(ValueError):
        build_train_transform(
            image_size=224,
            config=AugmentationConfig(random_resized_crop_scale_min=0.0),
        )


def test_build_train_eval_transforms_returns_pair() -> None:
    train_tf, eval_tf = build_train_eval_transforms(
        image_size=224,
        config=AugmentationConfig(),
    )

    assert _transform_type_names(train_tf)
    assert _transform_type_names(eval_tf)
