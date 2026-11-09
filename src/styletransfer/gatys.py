"""Gatys et al. (2015) optimization baseline: the slow way to do style transfer.

There is no trained network here. We start from the content photo and nudge
its pixels directly, a few hundred times, until the same VGG16 judge used in
training is happy with both content and style. It works for any single image
but starts from scratch every time, which is why it is slow. The point of this
file is to measure that cost against the trained TransformNet.

Usage:
    python -m styletransfer.gatys --content photo.jpg --style starry_night
"""
import argparse
import time
from pathlib import Path

import torch

from .data import load_image, save_image
from .losses import content_loss, gram_matrix, style_loss, tv_loss
from .train import load_config, style_targets
from .vgg import VGGFeatures


def gatys_stylize(
    content: torch.Tensor,
    targets: dict[str, torch.Tensor],
    style_index: int,
    vgg: VGGFeatures,
    loss_cfg: dict,
    steps: int = 300,
    lr: float = 0.02,
) -> tuple[torch.Tensor, list[float]]:
    """Optimize the pixels of `content` (1, 3, H, W) toward one style.

    Uses the same loss terms and weights as training, so the comparison is
    fair: only the method differs, not what "good" means.
    Returns the stylized image and the total loss at every step.
    """
    content_layer = loss_cfg["content_layer"]
    style_layers = loss_cfg["style_layers"]
    style_grams = [targets[layer][style_index : style_index + 1] for layer in style_layers]

    with torch.no_grad():
        content_feats = vgg(content)[content_layer]

    # Starting from the photo (not noise) keeps the layout and converges much faster.
    img = content.clone().requires_grad_(True)
    opt = torch.optim.Adam([img], lr=lr)
    history = []

    for _ in range(steps):
        feats = vgg(img)
        loss = (
            loss_cfg["content_weight"] * content_loss(feats[content_layer], content_feats)
            + loss_cfg["style_weight"] * style_loss([gram_matrix(feats[layer]) for layer in style_layers], style_grams)
            + loss_cfg["tv_weight"] * tv_loss(img)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # Keep pixels in the valid range; the TransformNet gets this from its sigmoid.
        with torch.no_grad():
            img.clamp_(0, 1)
        history.append(loss.item())

    return img.detach(), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--content", required=True, help="path to the photo to stylize")
    parser.add_argument("--style", required=True, help="style name from the config, e.g. starry_night")
    parser.add_argument("--size", type=int, default=512, help="short side of the photo, in pixels")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02, help="how far pixels move per step (0.02 gave the lowest loss of 0.02/0.05/0.1)")
    parser.add_argument("--style-weight", type=float, help="override the config's style weight")
    parser.add_argument("--out", help="output image path (default: runs/gatys/<photo>_<style>.jpg)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.style_weight:
        cfg["loss"]["style_weight"] = args.style_weight
    names = [s["name"] for s in cfg["styles"]]
    if args.style not in names:
        raise SystemExit(f"unknown style {args.style!r}; choose from {names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lcfg = cfg["loss"]
    layers = sorted(set(lcfg["style_layers"]) | {lcfg["content_layer"]})
    vgg = VGGFeatures(layers).to(device)
    targets = style_targets(vgg, cfg["styles"], cfg["style_size"], lcfg["style_layers"], device)
    content = load_image(args.content, args.size).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result, history = gatys_stylize(content, targets, names.index(args.style), vgg, lcfg, args.steps, args.lr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    out = Path(args.out) if args.out else Path("runs/gatys") / f"{Path(args.content).stem}_{args.style}.jpg"
    save_image(result, out)
    print(f"{args.steps} steps in {seconds:.1f} s, loss {history[0]:.0f} -> {history[-1]:.0f}, saved {out}")


if __name__ == "__main__":
    main()
