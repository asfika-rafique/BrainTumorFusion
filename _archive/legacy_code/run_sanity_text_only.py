# scripts/2_sanity_text_only.py
from pathlib import Path
import yaml, subprocess, sys, glob, os

BASE_CFG = "src/configs/debug.yaml"
TMP_CFG  = "src/configs/_sanity_txt.yaml"

def latest_ckpt():
    cks = glob.glob("outputs/checkpoints/*.pt")
    if not cks:
        print("No checkpoint found in outputs/checkpoints"); sys.exit(1)
    return max(cks, key=os.path.getmtime)

def main():
    with open(BASE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # text-only (ensure captions.csv exists)
    cfg["use_images"] = False
    cfg["use_text"]   = True

    if not Path(cfg["paths"]["captions_csv"]).exists():
        print(f"Missing captions CSV at {cfg['paths']['captions_csv']}")
        sys.exit(1)

    Path(TMP_CFG).parent.mkdir(parents=True, exist_ok=True)
    with open(TMP_CFG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    ckpt = latest_ckpt()
    print(f"[run] evaluate text-only with {ckpt}")
    subprocess.run([sys.executable, "-m", "scripts.evaluate",
                    "--cfg", TMP_CFG, "--ckpt", ckpt], check=True)

if __name__ == "__main__":
    main()
