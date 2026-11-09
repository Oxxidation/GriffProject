import torch

from styletransfer.gatys import gatys_stylize
from styletransfer.losses import gram_matrix
from styletransfer.vgg import VGGFeatures

LOSS_CFG = {
    "content_layer": "relu2_2",
    "style_layers": ["relu1_2", "relu2_2"],
    "content_weight": 1.0e5,
    "style_weight": 1.0e10,
    "tv_weight": 1.0e-6,
}


def test_gatys_reduces_loss_and_stays_in_pixel_range():
    torch.manual_seed(0)
    vgg = VGGFeatures(["relu1_2", "relu2_2"])
    style = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        feats = vgg(style)
    targets = {layer: gram_matrix(feats[layer]) for layer in LOSS_CFG["style_layers"]}

    content = torch.rand(1, 3, 64, 64)
    out, history = gatys_stylize(content, targets, 0, vgg, LOSS_CFG, steps=30)

    assert out.shape == content.shape
    assert out.min() >= 0 and out.max() <= 1
    assert history[-1] < history[0]
