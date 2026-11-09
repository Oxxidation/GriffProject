"""Content images (COCO) and style images, plus the shared image transforms.

Every tensor in this project is RGB in [0, 1]. ImageNet normalization happens
only inside the VGG wrapper, so the TransformNet, the app, and saved images all
use the same plain pixel range.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: str | Path) -> list[Path]:
    """All image files in a folder, sorted so splits are reproducible."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def split_holdout(paths: list[Path], holdout: int) -> tuple[list[Path], list[Path]]:
    """Last `holdout` images are kept out of training for evaluation."""
    return paths[:-holdout], paths[-holdout:]


class ContentDataset(Dataset):
    """Random square crops of content photos, the input side of training."""

    def __init__(self, paths: list[Path], image_size: int = 256):
        self.paths = paths
        # Resize the short side first so the crop keeps most of the scene.
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        # COCO has a few grayscale images; convert so every tensor has 3 channels.
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def load_image(path: str | Path, size: int | None = None) -> torch.Tensor:
    """One image as a (1, 3, H, W) tensor in [0, 1], optionally resized on its short side."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = transforms.Resize(size)(img)
    return transforms.ToTensor()(img).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a (3, H, W) or (1, 3, H, W) tensor in [0, 1] to disk."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(tensor.detach().float().clamp(0, 1).cpu()).save(path)
