"""Measure the trained models and write every report figure and table.

Every number in the report should come from a run of this script, so the
output names the checkpoints it measured and when.

Usage:
    python -m styletransfer.eval
    python -m styletransfer.eval --models final=weights/final.pth mild=runs/multistyle_40k/checkpoint.pth

Writes to report/figures/:
    results.md          all tables in one readable page
    speed.csv           model speed at each resolution, plus Gatys
    heldout.csv         content and style scores on photos never used in training
    grid_<model>.jpg    6 held-out photos x 4 styles, one grid per model
    grid_gatys.jpg      the same grid made with the slow Gatys method
    method_comparison.jpg  one photo, every method
    zoom_comparison.jpg    the center of that photo enlarged, to show checkerboard artifacts in a row
    blend_strip.jpg     smooth blends between two styles
    loss_curves.png     training curves from every full training run
"""
import argparse
import csv
import datetime
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .data import list_images, load_image, split_holdout
from .gatys import gatys_stylize
from .infer import Stylizer
from .losses import content_loss, gram_matrix, style_loss
from .train import load_config, style_targets
from .vgg import VGGFeatures

# Models trained with different loss weights can't be compared on their own
# training scores, so every model is scored with the same fixed yardstick.
REF_CONTENT_WEIGHT = 1e5
REF_STYLE_WEIGHT = 1e10

TILE = 256
_SINGLES = "+".join(f"runs/single_{s}/checkpoint.pth" for s in ["starry_night", "great_wave", "the_scream", "composition_vii"])
DEFAULT_MODELS = [
    "final_3e10=weights/final.pth",
    "first_1e10=runs/multistyle_40k/checkpoint.pth",
    "strong_1e11=runs/style1e11_40k/checkpoint.pth",
    "transpose_3e10=runs/transpose_40k/checkpoint.pth",
    f"single_style_x4={_SINGLES}",
]


class SingleStyleSet:
    """Several one-style models that act like one multi-style model.

    Used for the "one shared network vs one network per style" ablation: each
    style is routed to its own model. Blending between styles isn't possible
    here, which is one of the things the shared network buys.
    """

    def __init__(self, paths: list[str], device: str | None = None):
        self.parts = [Stylizer(p, device) for p in paths]
        self.by_style = {st.styles[0]: st for st in self.parts}
        self.styles = list(self.by_style)
        self.device = self.parts[0].device
        self.config = self.parts[0].config
        self.paths = paths

    def style_vector(self, style_a: str, style_b: str | None = None, alpha: float = 0.0) -> str:
        if style_b not in (None, style_a) and alpha not in (0, 0.0):
            raise ValueError("separate single-style models can't blend")
        return style_a

    def stylize_tensor(self, x: torch.Tensor, style: str, strength: float = 1.0) -> torch.Tensor:
        st = self.by_style[style]
        return st.stylize_tensor(x, st.style_vector(style), strength)


def load_model(spec_path: str):
    paths = spec_path.split("+")
    return SingleStyleSet(paths) if len(paths) > 1 else Stylizer(paths[0])


def model_size(st, paths: list[str]) -> tuple[int, float]:
    """Parameter count and checkpoint size in MB (summed over a SingleStyleSet)."""
    parts = st.parts if isinstance(st, SingleStyleSet) else [st]
    params = sum(sum(p.numel() for p in s.model.parameters()) for s in parts)
    return params, sum(Path(p).stat().st_size for p in paths) / 1e6


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def to_array(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) tensor in [0, 1] -> HxWx3 uint8 array."""
    return (t[0].detach().float().clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def square_tile(arr: np.ndarray, size: int = TILE) -> Image.Image:
    """Resize the short side to `size` and center-crop, so grids line up."""
    img = Image.fromarray(arr)
    scale = size / min(img.size)
    img = img.resize((max(size, round(img.width * scale)), max(size, round(img.height * scale))), Image.LANCZOS)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))


def labeled_grid(rows: list[list[np.ndarray]], col_labels: list[str], row_labels: list[str] | None = None) -> Image.Image:
    """Tiles in rows, with a label strip on top (and on the left if row_labels given)."""
    font = ImageFont.load_default(size=18)
    head, side = 34, (150 if row_labels else 0)
    grid = Image.new("RGB", (side + TILE * len(col_labels), head + TILE * len(rows)), "white")
    draw = ImageDraw.Draw(grid)
    for c, label in enumerate(col_labels):
        draw.text((side + c * TILE + TILE // 2, head // 2), label, fill="black", font=font, anchor="mm")
    for r, row in enumerate(rows):
        if row_labels:
            draw.text((side // 2, head + r * TILE + TILE // 2), row_labels[r], fill="black", font=font, anchor="mm")
        for c, arr in enumerate(row):
            grid.paste(square_tile(arr), (side + c * TILE, head + r * TILE))
    return grid


def time_model(st: Stylizer, sizes: list[int], runs: int) -> dict[int, float]:
    """Median milliseconds per frame for a 16:9 webcam-shaped frame at each short side."""
    frame = torch.rand(1, 3, 720, 1280, device=st.device)
    style = st.style_vector(st.styles[0])
    out = {}
    for size in sizes:
        x = F.interpolate(frame, size=(size, round(size * 16 / 9)), mode="bilinear", align_corners=False)
        for _ in range(10):  # warm-up, not timed
            st.stylize_tensor(x, style)
        times = []
        for _ in range(runs):
            sync(st.device)
            start = time.perf_counter()
            st.stylize_tensor(x, style)
            sync(st.device)
            times.append((time.perf_counter() - start) * 1000)
        out[size] = statistics.median(times)
    return out


@torch.inference_mode()
def score(vgg: VGGFeatures, out: torch.Tensor, photo_feats: dict, targets: dict, style_index: int, lcfg: dict) -> tuple[float, float]:
    """Content and style score of one stylized image, on the fixed yardstick."""
    feats = vgg(out.float())
    c = content_loss(feats[lcfg["content_layer"]], photo_feats[lcfg["content_layer"]]).item()
    s = style_loss(
        [gram_matrix(feats[layer]) for layer in lcfg["style_layers"]],
        [targets[layer][style_index : style_index + 1] for layer in lcfg["style_layers"]],
    ).item()
    return REF_CONTENT_WEIGHT * c, REF_STYLE_WEIGHT * s


@torch.inference_mode()
def heldout_scores(st: Stylizer, paths: list[Path], vgg, targets, style_names, lcfg, size) -> dict[str, tuple[float, float]]:
    """Mean content and style score per style over the held-out photos."""
    sums = {name: [0.0, 0.0] for name in st.styles}
    for p in paths:
        x = load_image(p, size).to(st.device)
        photo_feats = vgg(x)
        for name in st.styles:
            y = st.stylize_tensor(x, st.style_vector(name))
            c, s = score(vgg, y, photo_feats, targets, style_names.index(name), lcfg)
            sums[name][0] += c
            sums[name][1] += s
    return {name: (c / len(paths), s / len(paths)) for name, (c, s) in sums.items()}


def loss_curves(out_path: Path) -> list[str]:
    """Plot content and (re-scaled) style loss for every full training run in runs/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = []
    for log in sorted(Path("runs").glob("*/log.csv")):
        with open(log, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 50:  # skip short trial runs
            continue
        cfg = load_config(log.parent / "config.yaml")
        # The log's style column is multiplied by that run's style weight;
        # divide it back out so runs with different weights share one scale.
        rescale = REF_STYLE_WEIGHT / cfg["loss"]["style_weight"]
        it = np.array([int(r["iteration"]) for r in rows])
        c = np.array([float(r["content"]) for r in rows])
        s = np.array([float(r["style"]) for r in rows]) * rescale
        # Each logged value is one batch of 8, so smooth over 10 log points.
        k = np.ones(10) / 10
        label = f"{log.parent.name} (style weight {cfg['loss']['style_weight']:.0e})"
        ax_c.plot(it[9:], np.convolve(c, k, "valid"), label=label)
        ax_s.plot(it[9:], np.convolve(s, k, "valid"), label=label)
        plotted.append(log.parent.name)
    for ax, title in [(ax_c, "Content loss (lower = closer to the photo)"), (ax_s, "Style loss at weight 1e10 (lower = closer to the painting)")]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training round")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    ax_s.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return plotted


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="data split and loss layers")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="label=checkpoint pairs")
    parser.add_argument("--out", default="report/figures")
    parser.add_argument("--heldout", type=int, help="how many held-out photos to score (default: all)")
    parser.add_argument("--grid-images", type=int, default=6)
    parser.add_argument("--gatys-steps", type=int, default=300)
    parser.add_argument("--speed-runs", type=int, default=100)
    parser.add_argument("--size", type=int, default=256, help="short side used for scores and grids")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    lcfg = cfg["loss"]
    style_names = [s["name"] for s in cfg["styles"]]

    models, model_paths = {}, {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        missing = [p for p in path.split("+") if not Path(p).exists()]
        if missing:
            print(f"skipping {label}: {', '.join(missing)} not found")
            continue
        models[label] = load_model(path)
        model_paths[label] = path.split("+")
    if not models:
        raise SystemExit("no checkpoints found")
    first = next(iter(models.values()))
    device = first.device
    styles = first.styles

    _, holdout = split_holdout(list_images(cfg["data"]["content_dir"]), cfg["data"]["holdout"])
    scored = holdout[: args.heldout] if args.heldout else holdout
    grid_paths = holdout[: args.grid_images]

    vgg = VGGFeatures(sorted(set(lcfg["style_layers"]) | {lcfg["content_layer"]})).to(device)
    targets = style_targets(vgg, cfg["styles"], cfg["style_size"], lcfg["style_layers"], device)

    # 1. Speed: measured first, before anything else has warmed or loaded the GPU.
    print("timing models...")
    sizes = [256, 512, 720]
    speeds = {label: time_model(st, sizes, args.speed_runs) for label, st in models.items()}

    # 2. Scores on held-out photos the models never trained on.
    print(f"scoring {len(scored)} held-out photos x {len(styles)} styles per model...")
    scores = {label: heldout_scores(st, scored, vgg, targets, style_names, lcfg, args.size) for label, st in models.items()}

    # 3. Gatys on the grid photos: gives its grid, its timing, and a fair
    #    same-photos score comparison. Uses the final model's own loss weights
    #    so both methods chase the same target.
    print(f"running Gatys on {len(grid_paths)} photos x {len(styles)} styles ({args.gatys_steps} steps each)...")
    gatys_lcfg = models[next(iter(models))].config["loss"]
    gatys_rows, gatys_secs = [], []
    gatys_scores = {name: [0.0, 0.0] for name in styles}
    model_on_grid = {label: {name: [0.0, 0.0] for name in styles} for label in models}
    for p in grid_paths:
        x = load_image(p, args.size).to(device)
        with torch.inference_mode():
            photo_feats = vgg(x)
        row = [to_array(x)]
        for name in styles:
            sync(device)
            start = time.perf_counter()
            y, _ = gatys_stylize(x, targets, style_names.index(name), vgg, gatys_lcfg, args.gatys_steps)
            sync(device)
            gatys_secs.append(time.perf_counter() - start)
            row.append(to_array(y))
            c, s = score(vgg, y, photo_feats, targets, style_names.index(name), lcfg)
            gatys_scores[name][0] += c / len(grid_paths)
            gatys_scores[name][1] += s / len(grid_paths)
            for label, st in models.items():
                c, s = score(vgg, st.stylize_tensor(x, st.style_vector(name)), photo_feats, targets, style_names.index(name), lcfg)
                model_on_grid[label][name][0] += c / len(grid_paths)
                model_on_grid[label][name][1] += s / len(grid_paths)
        gatys_rows.append(row)
    gatys_median = statistics.median(gatys_secs)

    # 4. Figures.
    print("drawing figures...")
    pretty = {"starry_night": "Starry Night", "great_wave": "Great Wave", "the_scream": "The Scream", "composition_vii": "Composition VII"}
    col_labels = ["Photo"] + [pretty.get(n, n.replace("_", " ").title()) for n in styles]
    labeled_grid(gatys_rows, col_labels).save(out / "grid_gatys.jpg", quality=92)
    model_rows = {}
    for label, st in models.items():
        rows = []
        for p in grid_paths:
            x = load_image(p, args.size).to(device)
            rows.append([to_array(x)] + [to_array(st.stylize_tensor(x, st.style_vector(n))) for n in styles])
        model_rows[label] = rows
        labeled_grid(rows, col_labels).save(out / f"grid_{label}.jpg", quality=92)

    labeled_grid(
        [gatys_rows[0]] + [model_rows[label][0] for label in models],
        col_labels,
        ["Gatys"] + list(models),
    ).save(out / "method_comparison.jpg", quality=92)

    # Zoomed crops: checkerboard artifacts are a few pixels wide and vanish
    # when the whole image is shrunk into a grid tile, so blow up the center.
    zoom_rows = []
    for label in models:
        row = []
        for arr in model_rows[label][0][1:]:
            h, w = arr.shape[:2]
            crop = arr[h // 2 - 32 : h // 2 + 32, w // 2 - 32 : w // 2 + 32]
            row.append(np.asarray(Image.fromarray(crop).resize((TILE, TILE), Image.NEAREST)))
        zoom_rows.append(row)
    labeled_grid(zoom_rows, col_labels[1:], list(models)).save(out / "zoom_comparison.jpg", quality=95)

    final = models[next(iter(models))]
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    x = load_image(grid_paths[0], args.size).to(device)
    pairs = [(styles[0], styles[1]), (styles[2], styles[3])]
    blend_rows = [[to_array(x)] + [to_array(final.stylize_tensor(x, final.style_vector(a, b, al))) for al in alphas] for a, b in pairs]
    labeled_grid(
        blend_rows,
        ["Photo"] + [f"alpha {a}" for a in alphas],
        [f"{a.split('_')[0]} -> {b.split('_')[0]}" for a, b in pairs],
    ).save(out / "blend_strip.jpg", quality=92)

    curves = loss_curves(out / "loss_curves.png")

    # 5. Tables.
    with open(out / "speed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "short_side_px", "median_ms", "fps"])
        for label, by_size in speeds.items():
            for size, ms in by_size.items():
                w.writerow([label, size, f"{ms:.2f}", f"{1000 / ms:.1f}"])
        w.writerow(["gatys", args.size, f"{gatys_median * 1000:.0f}", f"{1 / gatys_median:.2f}"])

    with open(out / "heldout.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "style", "content_score", "style_score", "photos"])
        for label, by_style in scores.items():
            for name, (c, s) in by_style.items():
                w.writerow([label, name, f"{c:.0f}", f"{s:.0f}", len(scored)])

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    first_label = next(iter(models))
    speedup = gatys_median * 1000 / speeds[first_label][args.size]
    md = [
        "# Evaluation results",
        "",
        f"Generated by `python -m styletransfer.eval` on {datetime.datetime.now():%Y-%m-%d %H:%M}, on {gpu}.",
        "",
        "Models measured:",
        "",
        md_table(["Label", "Checkpoint", "Upsampling", "Style weight", "Training rounds", "Parameters", "File size"],
                 [[label,
                   "<br>".join(f"`{p}`" for p in model_paths[label]),
                   st.config["model"]["upsample"],
                   f"{st.config['loss']['style_weight']:.0e}",
                   f"{st.config['train']['iterations']:,}" + (" each" if isinstance(st, SingleStyleSet) else ""),
                   f"{model_size(st, model_paths[label])[0]:,}",
                   f"{model_size(st, model_paths[label])[1]:.1f} MB"]
                  for label, st in models.items()]),
        "",
        "## Speed",
        "",
        f"Median of {args.speed_runs} timed runs after 10 warm-up runs, one 16:9 frame, bf16 on the GPU. "
        f"Gatys: median of {len(gatys_secs)} images, {args.gatys_steps} steps each, at a {args.size} px short side.",
        "",
        md_table(["Method", "256 px", "512 px", "720 px"],
                 [[label, *[f"{by[s]:.1f} ms ({1000 / by[s]:.0f} fps)" for s in sizes]] for label, by in speeds.items()]
                 + [["Gatys", f"{gatys_median:.2f} s per image", "", ""]]),
        "",
        f"At {args.size} px the `{first_label}` model is **{speedup:,.0f}x faster** than Gatys.",
        "",
        "## Scores on held-out photos",
        "",
        f"Average over {len(scored)} COCO photos that were never used in training, at a {args.size} px short side. "
        f"Lower is better. Every model is scored with the same weights (content {REF_CONTENT_WEIGHT:.0e}, style {REF_STYLE_WEIGHT:.0e}) "
        "so models trained with different weights can be compared. Content measures how far the result drifted from the photo; "
        "style measures how far it is from the painting. The two pull against each other.",
        "",
        md_table(["Model"] + [f"{n} content / style" for n in styles] + ["Average content / style"],
                 [[label] + [f"{by[n][0]:,.0f} / {by[n][1]:,.0f}" for n in styles]
                  + [f"{np.mean([by[n][0] for n in styles]):,.0f} / {np.mean([by[n][1] for n in styles]):,.0f}"]
                  for label, by in scores.items()]),
        "",
        "## Gatys vs trained models on the same photos",
        "",
        f"The {len(grid_paths)} grid photos only (Gatys is too slow to run on all {len(scored)}). Same yardstick as above.",
        "",
        md_table(["Method", "Average content", "Average style"],
                 [["Gatys", f"{np.mean([gatys_scores[n][0] for n in styles]):,.0f}", f"{np.mean([gatys_scores[n][1] for n in styles]):,.0f}"]]
                 + [[label, f"{np.mean([by[n][0] for n in styles]):,.0f}", f"{np.mean([by[n][1] for n in styles]):,.0f}"]
                    for label, by in model_on_grid.items()]),
        "",
        "## Figures",
        "",
        "- `grid_gatys.jpg`, " + ", ".join(f"`grid_{label}.jpg`" for label in models) + f": {len(grid_paths)} held-out photos in every style",
        "- `method_comparison.jpg`: one photo, every method",
        "- `zoom_comparison.jpg`: the center of that photo enlarged 4x, to show fine artifacts such as checkerboard patterns",
        f"- `blend_strip.jpg`: blending between two styles with the `{first_label}` model",
        f"- `loss_curves.png`: training curves for {', '.join(curves) or 'no runs found'}",
        "",
    ]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"done, wrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
