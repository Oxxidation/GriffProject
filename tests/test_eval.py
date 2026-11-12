import pytest
import torch

from styletransfer.eval import SingleStyleSet, model_size
from styletransfer.model import TransformNet

CFG = {
    "model": {"channels": [8, 16, 32], "num_residual": 1, "upsample": "nearest"},
    "loss": {"style_weight": 1.0e10},
    "train": {"iterations": 10},
}


def save_single(tmp_path, style: str, seed: int) -> str:
    torch.manual_seed(seed)
    net = TransformNet(1, (8, 16, 32), 1)
    path = tmp_path / f"{style}.pth"
    torch.save({"model": net.state_dict(), "config": CFG, "styles": [style], "iteration": 10}, path)
    return str(path)


def test_single_style_set_routes_each_style_to_its_own_model(tmp_path):
    paths = [save_single(tmp_path, "a", 0), save_single(tmp_path, "b", 1)]
    combo = SingleStyleSet(paths, device="cpu")
    x = torch.rand(1, 3, 32, 32)

    assert combo.styles == ["a", "b"]
    a = combo.stylize_tensor(x, combo.style_vector("a"))
    direct_a = combo.parts[0].stylize_tensor(x, combo.parts[0].style_vector("a"))
    assert torch.equal(a, direct_a)
    assert not torch.equal(a, combo.stylize_tensor(x, combo.style_vector("b")))


def test_single_style_set_cannot_blend(tmp_path):
    combo = SingleStyleSet([save_single(tmp_path, "a", 0), save_single(tmp_path, "b", 1)], device="cpu")
    with pytest.raises(ValueError):
        combo.style_vector("a", "b", 0.5)


def test_model_size_sums_all_parts(tmp_path):
    paths = [save_single(tmp_path, "a", 0), save_single(tmp_path, "b", 1)]
    combo = SingleStyleSet(paths, device="cpu")
    one_params, one_mb = model_size(combo.parts[0], paths[:1])
    params, mb = model_size(combo, paths)
    assert params == 2 * one_params
    assert mb == pytest.approx(2 * one_mb, rel=0.01)
