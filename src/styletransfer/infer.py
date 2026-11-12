"""Load a trained TransformNet once and stylize images with it.

This is the one piece of code the app, the webcam script, and eval.py all
share, so "how an image gets stylized" is defined in exactly one place.

Usage:
    python -m styletransfer.infer --input photo.jpg --style starry_night
    python -m styletransfer.infer --input photo.jpg --style starry_night --blend-with great_wave --alpha 0.5
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import TransformNet

DEFAULT_CHECKPOINT = "weights/final.pth"


class Stylizer:
    """Holds the model on the GPU and turns images into stylized images."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # weights_only=True refuses to run code hidden in a checkpoint file; ours
        # only holds tensors, strings, and numbers, so it loads fine.
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.config: dict = ckpt["config"]  # the exact settings this model was trained with
        m = self.config["model"]
        self.styles: list[str] = ckpt["styles"]
        self.model = TransformNet(len(self.styles), tuple(m["channels"]), m["num_residual"], m["upsample"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        # channels_last lets the GPU's tensor cores run the convolutions faster.
        self.model.to(memory_format=torch.channels_last)
        self.use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.last_ms = 0.0

    def warmup(self, sizes: tuple[int, ...] = (256, 384, 512, 720)) -> None:
        """Run throwaway images so the first real one isn't slow.

        The first call on a fresh process takes ~340 ms on the RTX 5080 while
        CUDA loads its kernels; after that it is a few ms. Without this, the
        first number a demo viewer sees would be wrong.
        """
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for size in sizes:
            for _ in range(2):
                self.stylize(dummy, self.styles[0], size=size)

    def style_vector(self, style_a: str, style_b: str | None = None, alpha: float = 0.0) -> torch.Tensor:
        """Weights over all styles: (1 - alpha) of style A plus alpha of style B.

        The network mixes each style's scale and shift vectors with these
        weights, so alpha 0 is exactly style A and alpha 1 is exactly style B.
        """
        w = torch.zeros(1, len(self.styles), device=self.device)
        w[0, self.styles.index(style_a)] += 1.0 - alpha
        w[0, self.styles.index(style_b or style_a)] += alpha
        return w

    @torch.inference_mode()
    def stylize_tensor(self, x: torch.Tensor, style: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """x: (1, 3, H, W) in [0, 1] on self.device. Returns the same shape."""
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
            y = self.model(x, style).float()
        # Strength mixes the painting back with the original photo.
        return (1.0 - strength) * x + strength * y

    def stylize(
        self,
        image: Image.Image | np.ndarray,
        style_a: str,
        style_b: str | None = None,
        alpha: float = 0.0,
        strength: float = 1.0,
        size: int | None = 512,
    ) -> np.ndarray:
        """Stylize a PIL image or HxWx3 uint8 array. Returns an HxWx3 uint8 array.

        `size` is the short side the network sees. Smaller is faster; the
        caller can scale the result back up for display.
        """
        # np.array (not asarray) copies, because PIL hands back read-only memory
        # that PyTorch warns about.
        arr = np.array(image.convert("RGB") if isinstance(image, Image.Image) else image)
        x = torch.from_numpy(arr).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if size is not None and min(x.shape[-2:]) != size:
            h, w = x.shape[-2:]
            scale = size / min(h, w)
            x = F.interpolate(x, size=(round(h * scale), round(w * scale)), mode="bilinear", align_corners=False, antialias=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        y = self.stylize_tensor(x, self.style_vector(style_a, style_b, alpha), strength)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - start) * 1000

        return (y[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--blend-with", help="second style to blend toward")
    parser.add_argument("--alpha", type=float, default=0.5, help="0 = only --style, 1 = only --blend-with")
    parser.add_argument("--strength", type=float, default=1.0, help="0 = original photo, 1 = full painting")
    parser.add_argument("--size", type=int, default=512, help="short side in pixels")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out")
    args = parser.parse_args()

    stylizer = Stylizer(args.checkpoint)
    alpha = args.alpha if args.blend_with else 0.0
    out = stylizer.stylize(Image.open(args.input), args.style, args.blend_with, alpha, args.strength, args.size)

    name = args.style + (f"+{args.blend_with}{alpha:.2f}" if args.blend_with else "")
    path = Path(args.out) if args.out else Path("runs/infer") / f"{Path(args.input).stem}_{name}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(path)
    print(f"saved {path} ({out.shape[1]}x{out.shape[0]}, {stylizer.last_ms:.1f} ms incl. first-call warm-up)")


if __name__ == "__main__":
    main()
