import numpy as np
import pytest
import torch

from styletransfer.infer import Stylizer
from styletransfer.model import TransformNet

STYLES = ["a", "b", "c"]


@pytest.fixture
def stylizer(tmp_path) -> Stylizer:
    """A small random model saved in the same format train.py uses."""
    torch.manual_seed(0)
    net = TransformNet(len(STYLES), (8, 16, 32), 1)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith(("gamma", "beta")):
                p.add_(torch.randn_like(p) * 0.5)
    cfg = {"model": {"channels": [8, 16, 32], "num_residual": 1, "upsample": "nearest"}}
    path = tmp_path / "ckpt.pth"
    torch.save({"model": net.state_dict(), "config": cfg, "styles": STYLES, "iteration": 0}, path)
    return Stylizer(path, device="cpu")


def photo(h: int = 48, w: int = 64) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_output_is_uint8_image_at_requested_size(stylizer):
    out = stylizer.stylize(photo(48, 64), "a", size=32)
    assert out.dtype == np.uint8
    assert out.shape == (32, 43, 3)  # short side 48 -> 32, long side scaled to match


def test_size_none_keeps_original_size(stylizer):
    assert stylizer.stylize(photo(48, 64), "a", size=None).shape == (48, 64, 3)


def test_strength_zero_returns_original(stylizer):
    img = photo()
    out = stylizer.stylize(img, "a", strength=0.0, size=None)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_blend_endpoints_match_pure_styles(stylizer):
    img = photo()
    a = stylizer.stylize(img, "a", size=None)
    b = stylizer.stylize(img, "b", size=None)
    assert np.array_equal(stylizer.stylize(img, "a", "b", alpha=0.0, size=None), a)
    assert np.array_equal(stylizer.stylize(img, "a", "b", alpha=1.0, size=None), b)
    assert not np.array_equal(a, b)


def test_unknown_style_raises(stylizer):
    with pytest.raises(ValueError):
        stylizer.stylize(photo(), "not_a_style")
