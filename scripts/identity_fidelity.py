"""Experiment 1 - Identity-preservation fidelity (Reviewer #3).

ArcFace cosine similarity between the original and the enhanced image,
reported per method as mean +/- std.

Unpaired DarkFace protocol: reference = original low-light input, so the
'input' row is 1.0 by construction (identity mapping).  Optional paired
variant: --originals pointing at ground truth (e.g. LOL high/ or FiveK
expert_c test/) measures 'enhanced vs GT'.

Layout expected:
  --originals      : directory of reference images
  --enhanced-root  : root with one subdirectory per method; images inside a
                     method directory must share the same stem as the
                     original (e.g. 1.png <-> 1.png)
  --labels-dir     : optional per-image face bbox .txt (same stem);
                     4-number line = x1 y1 x2 y2 (or x y w h), 5+ number
                     line = WIDER-Face ellipse (major minor angle cx cy)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biometric_common as bc


def parse_args():
    p = argparse.ArgumentParser(description="Identity-preservation fidelity evaluation")
    p.add_argument("--originals", required=True,
                   help="directory of original (reference) images")
    p.add_argument("--enhanced-root", required=True,
                   help="root containing one subdirectory per method")
    p.add_argument("--methods", nargs="+", default=None,
                   help="method names; 'input' = identity mapping")
    p.add_argument("--labels-dir", default=None,
                   help="optional face bbox .txt dir (same stem as original)")
    p.add_argument("--embed-cache", default=None, help="optional .npz cache of embeddings")
    p.add_argument("--pgdb-root", default=None,
                   help="cloned PGDB-GAN repo for auto-enhance")
    p.add_argument("--pgdb-weights", default=None,
                   help="PGDB-GAN checkpoint for auto-enhance")
    bc.add_common_args(p)
    return p.parse_args()


def read_bbox(label_file: Path):
    """Return (x1, y1, x2, y2) from a 4-number box line or a WIDER-Face
    ellipse line (major minor angle cx cy ...)."""
    try:
        text = label_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        nums = [float(v) for v in parts[:6]]
        if len(nums) == 4:
            x1, y1, x2, y2 = nums
            if x2 <= x1 or y2 <= y1:  # x y w h style
                x2, y2 = x1 + x2, y1 + y2
            return (x1, y1, x2, y2)
        if len(nums) >= 5:
            major, minor, _, cx, cy = nums[:5]
            r = max(major, minor)
            return (cx - r, cy - r, cx + r, cy + r)
    return None


def run_pgdb_if_needed(args, root: Path):
    out_dir = root / "PGDB-GAN"
    if any(out_dir.glob("*.*")):
        return
    if args.methods is not None and "PGDB-GAN" not in args.methods:
        return
    if not (args.pgdb_root and args.pgdb_weights):
        print("NOTE: PGDB-GAN enhanced images not found; place them under "
              f"{out_dir} or pass --pgdb-root/--pgdb-weights to auto-run.")
        return
    test_py = Path(args.pgdb_root) / "scripts" / "test.py"
    if not test_py.exists():
        test_py = Path(args.pgdb_root) / "test.py"
    if not test_py.exists():
        print(f"test.py not found under {args.pgdb_root}")
        return
    cmd = ["python", str(test_py),
           "--data_path_test_low", str(Path(args.originals)),
           "--model_test", str(Path(args.pgdb_weights)),
           "--save", str(out_dir), "--seed", "2", "--gpu", "0"]
    print("running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(Path(args.pgdb_root)))
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"PGDB-GAN auto-run failed: {exc}")
        print(f"Run the released test.py manually and place outputs under {out_dir}")
        return
    renamed = 0
    for f in bc.list_images(out_dir):
        if f.stem.endswith("_enhance"):
            target = f.with_name(f.stem[:-len("_enhance")] + f.suffix)
            if not target.exists():
                try:
                    os.replace(f, target)
                    renamed += 1
                except OSError:
                    pass
    if renamed:
        print(f"renamed {renamed} PGDB-GAN '*_enhance' outputs to plain stems")


def main():
    args = parse_args()
    originals = bc.list_images(Path(args.originals))
    if not originals:
        raise SystemExit(f"No images found in {args.originals}")
    root = Path(args.enhanced_root)
    methods = args.methods or sorted(d.name for d in root.iterdir() if d.is_dir())
    if args.dry_run:
        print(f"originals: {len(originals)}")
        for m in methods:
            if m != "input":
                print(f"  {m}: {len(bc.list_images_recursive(root / m))} images")
        return

    run_pgdb_if_needed(args, root)
    methods = args.methods or sorted(d.name for d in root.iterdir() if d.is_dir())

    emb = bc.Embedder(rec_name=args.rec_name, root=args.insightface_root,
                      device=args.device, use_mtcnn=args.use_mtcnn)
    cache = bc.load_embed_cache(args.embed_cache) if args.embed_cache else {}

    def embed_file(img_path: Path, box=None):
        key = str(img_path)
        if key in cache:
            return cache[key]
        img = cv2.imread(str(img_path))
        if img is None:
            cache[key] = None
            return None
        if box is not None:
            out = emb.embed_crop(img, tuple(int(v) for v in box))
        else:
            out = emb.embed_bgr(img)
        cache[key] = out
        return out

    ref_embs, ref_files, ref_boxes = [], [], []
    for img_path in originals:
        box = None
        if args.labels_dir:
            label = Path(args.labels_dir) / (img_path.stem + ".txt")
            if label.exists():
                box = read_bbox(label)
        e = embed_file(img_path, box)
        if e is not None:
            ref_embs.append(e)
            ref_files.append(img_path)
            ref_boxes.append(box)
    print(f"embeddings of originals: {len(ref_embs)}/{len(originals)}")

    rows = []
    for m in methods:
        sims = []
        if m == "input":
            sims = [1.0] * len(ref_embs)
        else:
            mdir = root / m
            by_stem = {}
            for f in bc.list_images_recursive(mdir):
                by_stem[f.stem] = f
            for img_path, ref_emb, box in zip(ref_files, ref_embs, ref_boxes):
                cand = by_stem.get(img_path.stem)
                if cand is None:
                    continue
                e = embed_file(cand, box)
                if e is None:
                    continue
                sim = float(np.dot(np.asarray(ref_emb).reshape(-1), np.asarray(e).reshape(-1)))
                if np.isfinite(sim):
                    sims.append(sim)
        sims = np.asarray(sims, dtype=np.float64)
        row = {"method": m, "pairs": int(len(sims)),
               "mean_cosine": float(sims.mean()) if len(sims) else float("nan"),
               "std_cosine": float(sims.std(ddof=1)) if len(sims) > 1 else 0.0}
        rows.append(row)
        print(f"{m}: n={row['pairs']} mean={row['mean_cosine']:.4f} "
              f"std={row['std_cosine']:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "biometric_identity_fidelity.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("method,pairs,mean_cosine,std_cosine\n")
        for row in rows:
            fh.write(f"{row['method']},{row['pairs']},"
                     f"{row['mean_cosine']:.6f},{row['std_cosine']:.6f}\n")
    if args.embed_cache:
        bc.save_embed_cache(args.embed_cache, cache)
    bc.write_run_config(out_dir / "biometric_identity_fidelity_config.json", {
        "experiment": "identity_fidelity",
        "originals": str(Path(args.originals)),
        "enhanced_root": str(root),
        "methods": methods,
        "rec_model": args.rec_name,
        "mtcnn": args.use_mtcnn,
        "labels_dir": args.labels_dir,
        "n_originals": len(originals),
        "n_embedded": len(ref_embs),
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()