#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo", default="shuheikatoinfo/UtterTune-CosyVoice2-0.5B-ja-JSUTJVS"
    )
    ap.add_argument("--revision", default=None, help="e.g., v1.0.0")
    ap.add_argument("--out_dir", default="lora_weights/cv2/ja/jsutjvs")
    args = ap.parse_args()

    snapshot_download(args.repo, revision=args.revision, local_dir=args.out_dir)
    print(f"Downloaded to {args.out}/")
