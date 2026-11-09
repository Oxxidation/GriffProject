"""Content, Gram-matrix style, and total variation losses."""
import torch
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Channel-by-channel correlations of a feature map, normalized by C*H*W.

    Gram matrices throw away *where* features appear and keep *which* features
    appear together, which is why they capture style (texture, brush strokes)
    rather than layout. Computed in fp32 because bf16 sums over large H*W lose
    too much precision.
    """
    b, c, h, w = features.shape
    f = features.float().reshape(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def content_loss(output_feats: torch.Tensor, content_feats: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output_feats.float(), content_feats.float())


def style_loss(output_grams: list[torch.Tensor], target_grams: list[torch.Tensor]) -> torch.Tensor:
    """Sum of Gram MSEs over layers. Targets may have batch 1 and are broadcast."""
    loss = output_grams[0].new_zeros(())
    for g, t in zip(output_grams, target_grams):
        loss = loss + F.mse_loss(g, t.expand_as(g))
    return loss


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Total variation: penalizes neighboring-pixel differences to keep output smooth."""
    dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().sum()
    dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().sum()
    return dh + dw
