"""Gradio app: restyle photos or a live webcam feed with the trained model.

Run from the project folder:
    python app/app.py
    python app/app.py --checkpoint runs/my_model/checkpoint.pth   (use your own model)
then open http://127.0.0.1:7860 in a browser.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch

# Make `styletransfer` importable and relative paths (config, styles, weights)
# work no matter which folder the app is started from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from styletransfer.infer import DEFAULT_CHECKPOINT, Stylizer  # noqa: E402

RESOLUTIONS = {"256": 256, "384": 384, "512": 512, "720p": 720}
PRETTY = {
    "starry_night": "The Starry Night (Van Gogh)",
    "great_wave": "The Great Wave (Hokusai)",
    "the_scream": "The Scream (Munch)",
    "composition_vii": "Composition VII (Kandinsky)",
}

# Load once at startup and keep on the GPU; loading per request would cost
# far more than the stylizing itself.
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="trained model to load")
# parse_known_args ignores flags meant for other tools (e.g. when imported by tests).
args, _ = parser.parse_known_args()
stylizer = Stylizer(args.checkpoint)
stylizer.warmup(tuple(RESOLUTIONS.values()))
style_choices = [(PRETTY.get(s, s), s) for s in stylizer.styles]


def side_by_side(original: np.ndarray, stylized: np.ndarray) -> np.ndarray:
    """Original (resized to match) next to the stylized image."""
    small = stylizer.stylize(original, stylizer.styles[0], strength=0.0, size=min(stylized.shape[:2]))
    return np.concatenate([small, stylized], axis=1)


def readout(ms: float, size: int, extra: str = "") -> str:
    return f"**Model time:** {ms:.1f} ms per frame ({1000 / max(ms, 1e-6):.0f} fps possible) at {size} px{extra}"


def stylize_photo(image, style_a, style_b, alpha, strength, resolution, compare):
    if image is None:
        return None, "Upload a photo to start."
    size = RESOLUTIONS[resolution]
    out = stylizer.stylize(image, style_a, style_b, alpha, strength, size)
    ms = stylizer.last_ms  # read before side_by_side, which runs the model again
    if compare:
        out = side_by_side(image, out)
    return out, readout(ms, size)


class FrameClock:
    """Measures real end-to-end frames per second for the webcam stream.

    Model time alone is a few milliseconds; the browser round trip (encode,
    send, decode) is usually the real limit, so show both.
    """

    def __init__(self):
        self.times: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self.times = [t for t in self.times if now - t < 2.0] + [now]
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


clock = FrameClock()


def stylize_webcam(frame, style_a, style_b, alpha, strength, resolution, compare):
    if frame is None:
        return None, ""
    size = RESOLUTIONS[resolution]
    out = stylizer.stylize(frame, style_a, style_b, alpha, strength, size)
    ms = stylizer.last_ms  # read before side_by_side, which runs the model again
    if compare:
        out = side_by_side(frame, out)
    return out, readout(ms, size, f" · **Live stream:** {clock.tick():.0f} fps end to end")


# The Gatys baseline needs VGG16 (~530 MB), so load it only if someone uses that tab.
_gatys = {}


def run_gatys(image, style, steps):
    if image is None:
        return None, "Upload a photo first."
    from styletransfer.gatys import gatys_stylize
    from styletransfer.train import style_targets
    from styletransfer.vgg import VGGFeatures

    if not _gatys:
        # Use the loaded model's own training settings so both methods chase the same target.
        cfg = stylizer.config
        lcfg = cfg["loss"]
        layers = sorted(set(lcfg["style_layers"]) | {lcfg["content_layer"]})
        vgg = VGGFeatures(layers).to(stylizer.device)
        targets = style_targets(vgg, cfg["styles"], cfg["style_size"], lcfg["style_layers"], stylizer.device)
        _gatys.update(cfg=cfg, vgg=vgg, targets=targets, names=[s["name"] for s in cfg["styles"]])

    # Same 256 px short side the fast model is shown at, for a fair comparison.
    content = stylizer.stylize(image, style, strength=0.0, size=256)
    x = torch.from_numpy(content).to(stylizer.device).permute(2, 0, 1).unsqueeze(0).float() / 255

    if stylizer.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    y, _ = gatys_stylize(x, _gatys["targets"], _gatys["names"].index(style), _gatys["vgg"], _gatys["cfg"]["loss"], int(steps))
    if stylizer.device.type == "cuda":
        torch.cuda.synchronize()
    slow = time.perf_counter() - start

    fast = stylizer.stylize(image, style, size=256)
    fast_ms = stylizer.last_ms
    gatys_img = (y[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    both = np.concatenate([gatys_img, fast], axis=1)
    msg = (
        f"**Left, Gatys (slow method):** {slow:.2f} s for {int(steps)} steps  \n"
        f"**Right, trained model:** {fast_ms:.1f} ms  \n"
        f"The trained model was about **{slow * 1000 / max(fast_ms, 1e-6):,.0f}x faster** on this image."
    )
    return both, msg


with gr.Blocks(title="Real-Time Style Transfer") as demo:
    gr.Markdown(
        "# Real-Time Style Transfer\n"
        "One neural network, four painting styles. Pick a style, or blend two, and restyle a photo or your webcam."
    )

    with gr.Row():
        style_a = gr.Dropdown(style_choices, value=stylizer.styles[0], label="Style A")
        style_b = gr.Dropdown(style_choices, value=stylizer.styles[1], label="Style B")
    with gr.Row():
        alpha = gr.Slider(0, 1, value=0, step=0.05, label="Blend (0 = only Style A, 1 = only Style B)")
        strength = gr.Slider(0, 1, value=1, step=0.05, label="Strength (0 = original photo, 1 = full painting)")
    with gr.Row():
        resolution = gr.Radio(list(RESOLUTIONS), value="512", label="Resolution (short side, pixels)")
        compare = gr.Checkbox(value=False, label="Show original side by side")
    controls = [style_a, style_b, alpha, strength, resolution, compare]

    with gr.Tab("Photo"):
        with gr.Row():
            photo_in = gr.Image(type="numpy", sources=["upload", "clipboard"], label="Your photo")
            photo_out = gr.Image(type="numpy", label="Stylized", format="jpeg", interactive=False)
        photo_info = gr.Markdown()
        # Re-run on every change: the model is fast enough that sliders feel live.
        gr.on(
            [c.change for c in [photo_in, *controls]],
            stylize_photo,
            [photo_in, *controls],
            [photo_out, photo_info],
            show_progress="hidden",
        )

    with gr.Tab("Webcam"):
        gr.Markdown("Allow camera access in your browser, then press the record button to start the live feed.")
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
            cam_out = gr.Image(type="numpy", label="Live stylized", format="jpeg", interactive=False)
        cam_info = gr.Markdown()
        cam.stream(
            stylize_webcam,
            [cam, *controls],
            [cam_out, cam_info],
            stream_every=0.04,
            time_limit=600,
            concurrency_limit=1,
            show_progress="hidden",
        )

    with gr.Tab("Speed comparison"):
        gr.Markdown(
            "The classic method (Gatys et al., 2015) repaints one image by trial and error, starting from scratch "
            "every time. This runs it next to the trained model on the same photo and style."
        )
        with gr.Row():
            g_in = gr.Image(type="numpy", sources=["upload", "clipboard"], label="Photo")
            with gr.Column():
                g_style = gr.Dropdown(style_choices, value=stylizer.styles[0], label="Style")
                g_steps = gr.Slider(50, 500, value=300, step=50, label="Gatys steps")
                g_btn = gr.Button("Run both", variant="primary")
        g_out = gr.Image(type="numpy", label="Gatys (left) vs trained model (right)", interactive=False)
        g_info = gr.Markdown()
        g_btn.click(run_gatys, [g_in, g_style, g_steps], [g_out, g_info])


if __name__ == "__main__":
    demo.launch()
