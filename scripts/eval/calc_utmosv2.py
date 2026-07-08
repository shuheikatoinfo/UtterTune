#!/usr/bin/env python3
"""Compute UTMOSv2 naturalness (MOS) scores for a list of audio files.

Given a CSV of audio paths (--csv, headerless, one path per line), this
script runs UTMOSv2 on each file and writes `audio_path,utmos` rows to
--out. It does not compute any base/LoRA comparison or diff — that kind of
downstream aggregation, if needed, is left to a separate step.

Two pieces of engineering make this practical on large eval sets:
  * Resumable checkpointing: if --out already contains scores for some of
    the requested audio_paths, those rows are skipped on rerun, and results
    are checkpointed to --out after every chunk (--chunk_size) so a crash or
    interruption doesn't lose completed work.
  * OOM-safe batching: UTMOSv2 is run with --batch_size, but on CUDA OOM the
    batch size is automatically halved and retried (down to 1) rather than
    aborting the whole run.

Internally, audio files are symlinked into a flat directory (--link_dir) with
index-based filenames so they can be passed to UTMOSv2's batched predict()
API.
"""
import argparse
import csv
import os
from pathlib import Path
import sys

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch  # noqa: F401
    import utmosv2  # noqa: F401


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _prepare_symlink_dir(audio_paths: list[str], link_dir: Path) -> list[str]:
    """Create a flat directory of symlinks named by index: 000000.wav, ...

    Returns the list of stems (without .wav), in the same order as audio_paths.
    """

    link_dir.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []

    manifest_path = link_dir / "manifest_paths.txt"
    if manifest_path.is_file():
        existing = manifest_path.read_text(encoding="utf-8").splitlines()
    else:
        existing = []

    # If manifest differs, rebuild from scratch (safer than trying to diff).
    if existing != audio_paths:
        for p in link_dir.glob("*.wav"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        manifest_path.write_text("\n".join(audio_paths) + "\n", encoding="utf-8")

    for i, src in enumerate(audio_paths):
        stem = f"{i:06d}"
        dst = link_dir / f"{stem}.wav"
        stems.append(stem)
        # Path.exists() follows symlinks; a broken symlink would look "missing".
        # is_symlink() catches that case without following the link.
        if dst.is_symlink() or dst.exists():
            continue
        if not Path(src).is_file():
            raise FileNotFoundError(f"audio not found: {src}")
        dst.symlink_to(src)

    return stems


def _read_existing_scores(out_csv: Path, path_to_idx: dict[str, int]) -> dict[int, float]:
    if not out_csv.is_file():
        return {}

    scores: dict[int, float] = {}
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        if "audio_path" not in reader.fieldnames or "utmos" not in reader.fieldnames:
            return {}
        for row in reader:
            ap = (row.get("audio_path") or "").strip()
            if not ap:
                continue
            if ap not in path_to_idx:
                continue
            try:
                v = float(row.get("utmos", ""))
            except Exception:
                continue
            scores[path_to_idx[ap]] = v
    return scores


def _write_scores_checkpoint(out_csv: Path, audio_paths: list[str], score_by_idx: dict[int, float]) -> None:
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "utmos"])
        for i in sorted(score_by_idx.keys()):
            w.writerow([audio_paths[i], score_by_idx[i]])
    tmp.replace(out_csv)


def main(args):
    # If we use multiple DataLoader workers, numpy/librosa FFT can easily
    # oversubscribe CPU threads (workers * OMP threads). Cap threads to keep
    # throughput stable.
    num_workers = int(getattr(args, "num_workers", 0))
    cpu_threads = int(getattr(args, "cpu_threads", 0))
    if num_workers > 0 and cpu_threads <= 0:
        cpu_threads = 1
    if cpu_threads > 0:
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(k, str(cpu_threads))

    import torch
    import utmosv2

    # UTMOSv2 uses a DataLoader under the hood.
    # - `spawn` requires the dataset/args to be picklable (UTMOSv2 isn't).
    # - On Linux, `fork` usually works fine for CPU-side preprocessing.
    mp_start_method = getattr(args, "mp_start_method", None)
    if num_workers > 0 and mp_start_method and mp_start_method != "none":
        try:
            torch.multiprocessing.set_start_method(str(mp_start_method), force=True)
        except Exception:
            pass

    if cpu_threads > 0:
        try:
            torch.set_num_threads(cpu_threads)
        except Exception:
            pass

    # Input is a 1-column CSV (no header): audio_path
    audio_paths: list[str] = []
    with args.csv.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                audio_paths.append(s)

    path_to_idx = {p: i for i, p in enumerate(audio_paths)}
    score_by_idx: dict[int, float] = _read_existing_scores(args.out, path_to_idx)
    done = len(score_by_idx)
    total = len(audio_paths)

    if args.tqdm:
        print(f"[utmos] resume: {done}/{total} already scored", file=sys.stderr)

    remaining_indices = [i for i in range(total) if i not in score_by_idx]
    if not remaining_indices:
        if args.tqdm:
            print("[utmos] nothing to do (all scored)", file=sys.stderr)
        return

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = utmosv2.create_model(pretrained=True, device=args.device)

    # Use utmosv2's batched predict() by building a flat input_dir.
    link_dir = args.link_dir
    if link_dir is None:
        link_dir = args.out.parent / f"_{args.out.stem}_utmos_links"
    stems = _prepare_symlink_dir(audio_paths, link_dir)

    # Process remaining indices in chunks so we can checkpoint progress.
    chunk_size = max(1, int(args.chunk_size))

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    pbar = None
    if args.tqdm and tqdm is not None:
        pbar = tqdm(total=len(remaining_indices), desc="UTMOS", unit="clip", mininterval=5)

    # Aggressively batch for GPU utilization, but auto-backoff on OOM.
    batch_sizes: list[int] = []
    b = max(1, int(args.batch_size))
    while b >= 1:
        batch_sizes.append(b)
        if b == 1:
            break
        b = max(1, b // 2)

    for start in range(0, len(remaining_indices), chunk_size):
        chunk_indices = remaining_indices[start : start + chunk_size]
        chunk_stems = [stems[i] for i in chunk_indices]

        last_err: Exception | None = None
        results = None
        for bs in batch_sizes:
            try:
                with torch.inference_mode():
                    results = model.predict(
                        input_dir=link_dir,
                        val_list=chunk_stems,
                        device=args.device,
                        num_workers=int(args.num_workers),
                        batch_size=int(bs),
                        remove_silent_section=(not args.no_remove_silent_section),
                        verbose=False,
                    )
                if args.verbose:
                    print(f"[utmos] chunk ok (batch_size={bs}, n={len(chunk_stems)})", file=sys.stderr)
                break
            except RuntimeError as e:
                last_err = e
                if torch.cuda.is_available() and str(args.device).startswith("cuda") and _is_cuda_oom(e):
                    if args.verbose:
                        print(f"[utmos] OOM at batch_size={bs}; retrying smaller", file=sys.stderr)
                    if str(args.device).startswith("cuda"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                raise

        if results is None:
            raise SystemExit(f"UTMOS prediction failed: {last_err}")

        # `results` is list[{'file_path': str, 'predicted_mos': float}] for input_dir.
        for item in results:
            idx = int(Path(str(item["file_path"])).stem)
            score_by_idx[idx] = float(item["predicted_mos"])

        # checkpoint after each chunk (resume without redoing CPU)
        _write_scores_checkpoint(args.out, audio_paths, score_by_idx)

        if pbar is not None:
            pbar.update(len(chunk_indices))
        if args.tqdm:
            print(f"[utmos] checkpoint: {len(score_by_idx)}/{total}", file=sys.stderr)

    if pbar is not None:
        pbar.close()

    if args.tqdm:
        print(f"[utmos] done: {len(score_by_idx)}/{total}", file=sys.stderr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path, help="CSV with audio_path")
    p.add_argument("--out", type=Path, default="clip_scores.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true", help="Print per-clip MOS (very noisy for large CSVs)")
    p.add_argument("--batch_size", type=int, default=64, help="UTMOS batch size (auto-backoff on OOM)")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for audio loading")
    p.add_argument("--chunk_size", type=int, default=2048, help="How many clips to process per checkpoint chunk")
    p.add_argument("--tqdm", action="store_true", help="Show tqdm progress for chunked UTMOS computation")
    p.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="Cap CPU threads for FFT/BLAS (0=auto/no override; auto=1 when --num_workers>0)",
    )
    p.add_argument(
        "--mp_start_method",
        choices=["none", "fork", "spawn", "forkserver"],
        default="fork",
        help="Multiprocessing start method for DataLoader when --num_workers>0 (default: fork)",
    )
    p.add_argument(
        "--link_dir",
        type=Path,
        default=None,
        help="Directory to place symlinks for batched predict(); defaults next to --out",
    )
    p.add_argument(
        "--no_remove_silent_section",
        action="store_true",
        help="Disable silent-section trimming inside UTMOSv2 (faster but changes scoring behavior)",
    )
    main(p.parse_args())
