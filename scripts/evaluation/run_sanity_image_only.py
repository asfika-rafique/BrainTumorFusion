# scripts/evaluation/run_sanity_image_only.py
from pathlib import Path
import subprocess, sys

from scripts._common import ROOT

TMP_CFG = ROOT / "configs" / "sanity_image_only.yaml"

def latest_ckpt():
    cks = list((ROOT / "outputs" / "checkpoints").glob("*.pt"))
    if not cks:
        raise FileNotFoundError("No checkpoint found in outputs/checkpoints")
    return max(cks, key=lambda path: path.stat().st_mtime)

def main():
    ckpt = latest_ckpt()
    print(f"[run] evaluate image-only with {ckpt}")
    subprocess.run([sys.executable, "-m", "scripts.evaluation.evaluate",
                    "--cfg", str(TMP_CFG), "--ckpt", str(ckpt)], check=True)

if __name__ == "__main__":
    main()
