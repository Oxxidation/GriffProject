import torch

from styletransfer.model import TransformNet


def make_net(**kw) -> TransformNet:
    torch.manual_seed(0)
    net = TransformNet(num_styles=4, **kw)
    # Fresh nets have identical gamma/beta for every style; randomize them so
    # the blend tests can tell styles apart.
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("gamma") or name.endswith("beta"):
                p.add_(torch.randn_like(p) * 0.5)
    return net.eval()


def test_output_shape_matches_input_for_odd_sizes():
    net = make_net()
    for h, w in [(256, 256), (123, 77), (720, 1280)]:
        x = torch.rand(1, 3, h, w)
        y = net(x, torch.tensor([0]))
        assert y.shape == x.shape
        assert y.min() >= 0 and y.max() <= 1


def test_transpose_upsample_variant_runs():
    net = make_net(upsample="transpose")
    x = torch.rand(2, 3, 64, 64)
    assert net(x, torch.tensor([1, 2])).shape == x.shape


def test_blend_endpoints_equal_pure_styles():
    net = make_net()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        style_a = net(x, torch.tensor([0]))
        style_b = net(x, torch.tensor([2]))
        alpha0 = net(x, torch.tensor([[1.0, 0, 0, 0]]))
        alpha1 = net(x, torch.tensor([[0, 0, 1.0, 0]]))
        mid = net(x, torch.tensor([[0.5, 0, 0.5, 0]]))
    assert torch.allclose(alpha0, style_a, atol=1e-5)
    assert torch.allclose(alpha1, style_b, atol=1e-5)
    assert not torch.allclose(mid, style_a, atol=1e-3)
    assert not torch.allclose(style_a, style_b, atol=1e-3)
