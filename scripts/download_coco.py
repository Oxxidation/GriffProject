"""Download the COCO val2017 photos used as training content.

Usage:
    python scripts/download_coco.py

Downloads about 780 MB once, unpacks 5,000 photos into data/coco/val2017/,
and deletes the zip. Running it again skips the download if the photos are
already there, so it is safe to rerun.

The photos are not stored in the git repo: they are too large, and each one
belongs to its Flickr photographer under the license listed by COCO
(https://cocodataset.org/#termsofuse).

THIS ENTIRE THING IS WRITTEN BY CLAUDE SO YOU DON'T HAVE TO FIND THE DATASET YOURSELF
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://images.cocodataset.org/zips/val2017.zip"
EXPECTED_IMAGES = 5000

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "coco"
IMAGE_DIR = OUT_DIR / "val2017"
ZIP_PATH = OUT_DIR / "val2017.zip"


def count_images() -> int:
    if not IMAGE_DIR.is_dir():
        return 0
    return sum(1 for p in IMAGE_DIR.iterdir() if p.suffix.lower() == ".jpg")


def show_progress(blocks: int, block_size: int, total: int) -> None:
    done = blocks * block_size
    if total > 0:
        pct = min(100.0, done * 100 / total)
        print(f"\r  {done / 1024**2:6.0f} / {total / 1024**2:.0f} MB  ({pct:5.1f}%)", end="", flush=True)


def main() -> int:
    if count_images() >= EXPECTED_IMAGES:
        print(f"Already have {count_images()} photos in {IMAGE_DIR}. Nothing to do.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    # Download to a temporary name so an interrupted download is never
    # mistaken for a finished one on the next run.
    partial = ZIP_PATH.with_suffix(".zip.part")
    urllib.request.urlretrieve(URL, partial, show_progress)
    partial.replace(ZIP_PATH)
    print()

    print(f"Unpacking into {IMAGE_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(OUT_DIR)
    ZIP_PATH.unlink()

    n = count_images()
    if n < EXPECTED_IMAGES:
        print(f"Expected {EXPECTED_IMAGES} photos but found {n}. Delete {IMAGE_DIR} and run again.")
        return 1
    print(f"Done: {n} photos in {IMAGE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
