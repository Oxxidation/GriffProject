"""Train every ablation model, one after another.

Usage (from the project folder):
    python scripts/run_ablations.py

Runs, in this order (about 1 hour 45 minutes on an RTX 5080):
    1. Four single-style models, 10,000 rounds each. Our 4-style model gives
       each style about 10,000 rounds of practice, so this is a same-effort
       comparison: does sharing one network cost quality?
    2. Style weight 1e11, the third point on the content-vs-style trade-off
       (we already have 1e10 and 3e10).
    3. Transposed-convolution upsampling instead of resize-then-conv, to show
       the checkerboard artifacts that resize-then-conv avoids.

A run that already finished is skipped, so the script can be restarted after
an interruption. Each run's output goes to runs/<name>/ and runs/<name>.log.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STYLES = ["starry_night", "great_wave", "the_scream", "composition_vii"]

RUNS = [(f"single_{s}", ["--styles", s, "--iterations", "10000"]) for s in STYLES] + [
    ("style1e11_40k", ["--style-weight", "1e11"]),
    ("transpose_40k", ["--upsample", "transpose"]),
]


def finished(name: str) -> bool:
    log = ROOT / "runs" / f"{name}.log"
    return (ROOT / "runs" / name / "checkpoint.pth").exists() and log.exists() and "done in" in log.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    (ROOT / "runs").mkdir(exist_ok=True)
    for i, (name, extra) in enumerate(RUNS, 1):
        if finished(name):
            print(f"[{i}/{len(RUNS)}] {name}: already done, skipping", flush=True)
            continue
        print(f"[{i}/{len(RUNS)}] {name}: training...", flush=True)
        cmd = [sys.executable, "-m", "styletransfer.train", "--out", f"runs/{name}", *extra]
        with open(ROOT / "runs" / f"{name}.log", "w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"[{i}/{len(RUNS)}] {name}: FAILED, see runs/{name}.log", flush=True)
            return result.returncode
        print(f"[{i}/{len(RUNS)}] {name}: {(ROOT / 'runs' / f'{name}.log').read_text(encoding='utf-8').strip().splitlines()[-1]}", flush=True)
    print("all ablation runs finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
