"""TransformNet: a feed-forward network that stylizes an image in one pass.

Architecture follows Johnson et al. (2016): downsample, residual blocks,
upsample. Multi-style support follows Dumoulin et al. (2017): every style
shares the conv weights and gets its own scale (gamma) and shift (beta) in each
normalization layer. A style is "just" those vectors, which is why blending two
styles is a weighted average of them.
"""
import torch
from torch import nn


class ConditionalInstanceNorm(nn.Module):
    """Instance norm whose scale and shift depend on the chosen style.

    `style` is a (B, num_styles) weight matrix: a one-hot row picks a single
    style, and a row like [0.3, 0.7, 0, 0] blends two styles. Training always
    uses one-hot rows; blending only happens at inference.
    """

    def __init__(self, channels: int, num_styles: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.gamma = nn.Parameter(torch.ones(num_styles, channels))
        self.beta = nn.Parameter(torch.zeros(num_styles, channels))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma = (style @ self.gamma).unsqueeze(-1).unsqueeze(-1)
        beta = (style @ self.beta).unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * gamma + beta


class ConvLayer(nn.Module):
    """Reflection pad + conv. Reflection padding avoids dark borders that zero padding causes."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ConvNormReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, num_styles: int):
        super().__init__()
        self.conv = ConvLayer(in_ch, out_ch, kernel, stride)
        self.norm = ConditionalInstanceNorm(out_ch, num_styles)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x), style))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, num_styles: int):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3)
        self.norm1 = ConditionalInstanceNorm(channels, num_styles)
        self.conv2 = ConvLayer(channels, channels, 3)
        self.norm2 = ConditionalInstanceNorm(channels, num_styles)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.norm1(self.conv1(x), style))
        y = self.norm2(self.conv2(y), style)
        return x + y


class UpsampleConv(nn.Module):
    """2x upsample then conv.

    "nearest" (resize-then-conv) avoids the checkerboard artifacts that
    transposed convolutions produce when their kernel overlaps unevenly.
    "transpose" is kept only for the ablation that shows the difference.
    """

    def __init__(self, in_ch: int, out_ch: int, num_styles: int, mode: str = "nearest"):
        super().__init__()
        self.mode = mode
        if mode == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv = ConvLayer(in_ch, out_ch, 3)
        elif mode == "transpose":
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        else:
            raise ValueError(f"unknown upsample mode {mode!r}")
        self.norm = ConditionalInstanceNorm(out_ch, num_styles)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if self.mode == "nearest":
            x = self.up(x)
        return torch.relu(self.norm(self.conv(x), style))


class TransformNet(nn.Module):
    def __init__(
        self,
        num_styles: int,
        channels: tuple[int, int, int] = (32, 64, 128),
        num_residual: int = 5,
        upsample: str = "nearest",
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.num_styles = num_styles

        self.encoder = nn.ModuleList([
            ConvNormReLU(3, c1, 9, 1, num_styles),
            ConvNormReLU(c1, c2, 3, 2, num_styles),
            ConvNormReLU(c2, c3, 3, 2, num_styles),
        ])
        self.residuals = nn.ModuleList([ResidualBlock(c3, num_styles) for _ in range(num_residual)])
        self.decoder = nn.ModuleList([
            UpsampleConv(c3, c2, num_styles, upsample),
            UpsampleConv(c2, c1, num_styles, upsample),
        ])
        # Last layer has no norm, so the network can produce any color.
        self.out = ConvLayer(c1, 3, 9)

    def style_weights(self, style_ids: torch.Tensor) -> torch.Tensor:
        """Turn integer style ids (B,) into one-hot weight rows (B, num_styles)."""
        return nn.functional.one_hot(style_ids, self.num_styles).float()

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1]. style: int ids (B,) or weight rows (B, num_styles)."""
        if style.dim() == 1:
            style = self.style_weights(style)
        style = style.to(x.device, x.dtype if x.is_floating_point() else torch.float32)

        h, w = x.shape[-2:]
        y = x
        for layer in self.encoder:
            y = layer(y, style)
        for block in self.residuals:
            y = block(y, style)
        for layer in self.decoder:
            y = layer(y, style)
        y = self.out(y)

        # Two stride-2 layers mean sizes not divisible by 4 come back slightly
        # off; resize so any input size returns the same size.
        if y.shape[-2:] != (h, w):
            y = nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        # Sigmoid keeps output in the valid [0, 1] pixel range.
        return torch.sigmoid(y)
