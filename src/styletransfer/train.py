"""Train a TransformNet on one or more styles with perceptual losses.

Usage:
    python -m styletransfer.train --config configs/default.yaml
    python -m styletransfer.train --styles starry_night --iterations 1000 --out runs/tune
"""
import argparse
import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import ContentDataset, list_images, load_image, save_image, split_holdout
from .losses import content_loss, gram_matrix, style_loss, tv_loss
from .model import TransformNet
from .vgg import VGGFeatures


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # PyYAML follows YAML 1.1, which reads "1.0e5" (no exponent sign) as a
    # string. Cast the numeric fields so a config typo can't become a crash.
    for key in ("content_weight", "style_weight", "tv_weight"):
        cfg["loss"][key] = float(cfg["loss"][key])
    cfg["train"]["lr"] = float(cfg["train"]["lr"])
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infinite(loader: DataLoader):
    """Training is measured in iterations, not epochs, so loop the loader forever."""
    while True:
        yield from loader


@torch.no_grad()
def style_targets(vgg: VGGFeatures, styles: list[dict], size: int, layers: list[str], device) -> dict[str, torch.Tensor]:
    """Gram targets for every style, computed once up front.

    Returns {layer: (num_styles, C, C)} so a batch of style ids can index them.
    """
    grams = {layer: [] for layer in layers}
    for s in styles:
        feats = vgg(load_image(s["path"], size).to(device))
        for layer in layers:
            grams[layer].append(gram_matrix(feats[layer])[0])
    return {layer: torch.stack(g) for layer, g in grams.items()}


def build_model(cfg: dict, num_styles: int) -> TransformNet:
    m = cfg["model"]
    return TransformNet(num_styles, tuple(m["channels"]), m["num_residual"], m["upsample"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--styles", nargs="*", help="train only these style names (default: all)")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--style-weight", type=float)
    parser.add_argument("--upsample", choices=["nearest", "transpose"])
    parser.add_argument("--out")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Command-line overrides are written back into the config, so the copy
    # saved with each checkpoint records exactly what this run used.
    if args.styles:
        cfg["styles"] = [s for s in cfg["styles"] if s["name"] in args.styles]
        if len(cfg["styles"]) != len(args.styles):
            raise SystemExit(f"unknown style in {args.styles}")
    if args.iterations:
        cfg["train"]["iterations"] = args.iterations
    if args.style_weight:
        cfg["loss"]["style_weight"] = args.style_weight
    if args.upsample:
        cfg["model"]["upsample"] = args.upsample
    if args.out:
        cfg["train"]["out_dir"] = args.out

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg, lcfg = cfg["train"], cfg["loss"]
    out_dir = Path(tcfg["out_dir"])
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_paths, holdout_paths = split_holdout(list_images(cfg["data"]["content_dir"]), cfg["data"]["holdout"])
    dataset = ContentDataset(train_paths, cfg["data"]["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    print(f"{len(train_paths)} training images, {len(holdout_paths)} held out")

    styles = cfg["styles"]
    style_names = [s["name"] for s in styles]
    num_styles = len(styles)
    layers = sorted(set(lcfg["style_layers"]) | {lcfg["content_layer"]})
    vgg = VGGFeatures(layers).to(device)
    targets = style_targets(vgg, styles, cfg["style_size"], lcfg["style_layers"], device)

    net = build_model(cfg, num_styles).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg["lr"])
    amp = tcfg["amp"] and device.type == "cuda"

    # A fixed held-out photo, stylized in every style during training, shows
    # progress at a glance and catches bad loss weights early.
    preview = load_image(holdout_paths[0], cfg["data"]["image_size"]).to(device)

    log_file = open(out_dir / "log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["iteration", "content", "style", "tv", "total", "images_per_sec"])

    batches = infinite(loader)
    net.train()
    start = time.perf_counter()
    for it in range(1, tcfg["iterations"] + 1):
        x = next(batches).to(device, non_blocking=True)
        # Each image in the batch gets its own random style, so every step
        # trains every style's gamma/beta a little.
        ids = torch.randint(num_styles, (x.shape[0],), device=device)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            y = net(x, ids)
            feats_y = vgg(y)
            with torch.no_grad():
                feats_x = vgg(x)

        c_loss = lcfg["content_weight"] * content_loss(feats_y[lcfg["content_layer"]], feats_x[lcfg["content_layer"]])
        s_loss = lcfg["style_weight"] * style_loss(
            [gram_matrix(feats_y[layer]) for layer in lcfg["style_layers"]],
            [targets[layer][ids] for layer in lcfg["style_layers"]],
        )
        t_loss = lcfg["tv_weight"] * tv_loss(y.float())
        loss = c_loss + s_loss + t_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % tcfg["log_every"] == 0:
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.perf_counter() - start
            ips = it * tcfg["batch_size"] / elapsed
            log.writerow([it, f"{c_loss.item():.4f}", f"{s_loss.item():.4f}", f"{t_loss.item():.4f}", f"{loss.item():.4f}", f"{ips:.1f}"])
            log_file.flush()
            eta = (tcfg["iterations"] - it) * tcfg["batch_size"] / ips / 60
            print(f"it {it:6d}  content {c_loss.item():9.2f}  style {s_loss.item():9.2f}  tv {t_loss.item():7.2f}  {ips:6.1f} img/s  eta {eta:5.1f} min", flush=True)

        if it % tcfg["sample_every"] == 0 or it == tcfg["iterations"]:
            net.eval()
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                outs = [net(preview, torch.tensor([i], device=device)).float() for i in range(num_styles)]
            save_image(torch.cat([preview] + outs, dim=3), out_dir / "samples" / f"it{it:06d}.jpg")
            net.train()

        if it % tcfg["checkpoint_every"] == 0 or it == tcfg["iterations"]:
            ckpt = {"model": net.state_dict(), "config": copy.deepcopy(cfg), "styles": style_names, "iteration": it}
            torch.save(ckpt, out_dir / "checkpoint.pth")

    log_file.close()
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min, checkpoint at {out_dir / 'checkpoint.pth'}")


if __name__ == "__main__":
    main()
