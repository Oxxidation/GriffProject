import torch

from styletransfer.losses import content_loss, gram_matrix, style_loss, tv_loss


def test_gram_shape_and_symmetry():
    feats = torch.randn(2, 16, 8, 8)
    g = gram_matrix(feats)
    assert g.shape == (2, 16, 16)
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-6)


def test_gram_values_match_hand_computation():
    # One channel of all ones on a 2x2 map: sum of squares = 4, divided by C*H*W = 4.
    feats = torch.ones(1, 1, 2, 2)
    assert torch.allclose(gram_matrix(feats), torch.tensor([[[1.0]]]))


def test_gram_ignores_spatial_layout():
    # Style should not care where features are, so a shuffled map has the same Gram.
    feats = torch.randn(1, 4, 6, 6)
    perm = torch.randperm(36)
    shuffled = feats.reshape(1, 4, 36)[:, :, perm].reshape(1, 4, 6, 6)
    assert torch.allclose(gram_matrix(feats), gram_matrix(shuffled), atol=1e-5)


def test_losses_are_zero_for_identical_inputs():
    x = torch.rand(1, 3, 16, 16)
    assert content_loss(x, x) == 0
    assert style_loss([gram_matrix(x)], [gram_matrix(x)]) == 0
    assert tv_loss(torch.ones(1, 3, 8, 8)) == 0


def test_style_loss_decreases_on_toy_optimization():
    torch.manual_seed(0)
    target = gram_matrix(torch.rand(1, 3, 16, 16))
    img = torch.rand(1, 3, 16, 16, requires_grad=True)
    opt = torch.optim.Adam([img], lr=0.05)

    first = style_loss([gram_matrix(img)], [target]).item()
    for _ in range(100):
        opt.zero_grad()
        loss = style_loss([gram_matrix(img)], [target])
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.1
