from dataclasses import dataclass

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    hflip_prob: float = 0.5
    color_jitter_strength: float = 0.2
    rotation_degrees: float = 0.0
    random_resized_crop_scale_min: float = 1.0
    random_erasing_prob: float = 0.0


def _validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")


def _validate_config(config: AugmentationConfig) -> None:
    _validate_probability("hflip_prob", float(config.hflip_prob))
    _validate_probability("random_erasing_prob", float(config.random_erasing_prob))

    if config.color_jitter_strength < 0.0:
        raise ValueError("color_jitter_strength must be >= 0")
    if config.rotation_degrees < 0.0:
        raise ValueError("rotation_degrees must be >= 0")

    scale_min = float(config.random_resized_crop_scale_min)
    if scale_min <= 0.0 or scale_min > 1.0:
        raise ValueError("random_resized_crop_scale_min must be within (0, 1]")


def build_eval_transform(image_size: int) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be > 0")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(image_size: int, config: AugmentationConfig) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be > 0")

    _validate_config(config)

    if not config.enabled:
        return build_eval_transform(image_size)

    transform_steps = []

    if float(config.random_resized_crop_scale_min) < 1.0:
        transform_steps.append(
            transforms.RandomResizedCrop(
                (image_size, image_size),
                scale=(float(config.random_resized_crop_scale_min), 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    else:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    if float(config.hflip_prob) > 0.0:
        transform_steps.append(transforms.RandomHorizontalFlip(p=float(config.hflip_prob)))

    if float(config.rotation_degrees) > 0.0:
        transform_steps.append(
            transforms.RandomRotation(
                degrees=float(config.rotation_degrees),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    jitter = float(config.color_jitter_strength)
    if jitter > 0.0:
        transform_steps.append(
            transforms.ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=min(0.1, jitter * 0.5),
            )
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if float(config.random_erasing_prob) > 0.0:
        transform_steps.append(
            transforms.RandomErasing(
                p=float(config.random_erasing_prob),
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    return transforms.Compose(transform_steps)


def build_train_eval_transforms(
    image_size: int,
    config: AugmentationConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = build_train_transform(image_size=image_size, config=config)
    eval_tf = build_eval_transform(image_size=image_size)
    return train_tf, eval_tf
