"""Frozen VGG16 used only as a loss network (transfer learning).

VGG is never trained here. Its ImageNet features already capture edges,
textures, and object parts, which is exactly what content and style losses
need to compare.
"""
import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

# Index of each named ReLU inside torchvision's vgg16().features.
LAYER_INDEX = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_3": 15,
    "relu4_3": 22,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VGGFeatures(nn.Module):
    """Returns the requested VGG16 activations for images in [0, 1]."""

    def __init__(self, layers: list[str]):
        super().__init__()
        self.layers = layers
        last = max(LAYER_INDEX[name] for name in layers)
        # Cut the network after the deepest layer we need; the rest is wasted compute.
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: last + 1]
        self.index_to_name = {LAYER_INDEX[name]: name for name in layers}

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "VGGFeatures":
        # Always stay in eval mode, even if a parent module calls .train().
        return super().train(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.index_to_name:
                out[self.index_to_name[i]] = x
        return out
