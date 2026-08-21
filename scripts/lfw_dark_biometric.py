"""Experiments 2 & 3 - Controlled low-light recognition and verification on
LFW aligned. One pipeline, two statistic blocks.

prepare : LFW normal-light -> synthetic low-light (DEGRADE_PARAMS, fixed
          seed; per-image noise derived deterministically from the path).
          By default only the images referenced by --pairs-file are
          processed (View-2 fold 1 = about 1,100-1,300 images); pass
          --degrade-all for the full 13,233-image set.
enhance : synthetic dark -> per-method outputs ('input' = identity copy;
          PGDB-GAN optionally auto-run via the released test.py, including
          flatten/remap for the nested LFW layout)
evaluate: (2) verification TAR@FAR=1%/ROC AUC from the given View-2 pairs
          (probe = enhanced image of the first pair member vs the
          normal-light second member; TAR@FAR=0.1% is below the resolution
          of a 300-impostor fold and prints as nan);
          (3) recognition Rank-1/5 (gallery = normal-light LFW, probe =
          enhanced dark images)

Example (fast, official fold 1):
 python lfw_dark_biometric.py --lfw <lfw_aligned> --pairs-file <pairs_fold1.txt> \
   --dark-root lfw_dark --enhanced-root enhanced \
   --methods input SCI SNR-Net Retinexformer PGDB-GAN \
   --pgdb-root <repo> --pgdb-weights <weights/LOL.pt>

Example (full View-2, 6,000 pairs):
 python lfw_dark_biometric.py --lfw <lfw_aligned> --pairs-file <pairs.txt> \
   --degrade-all --dark-root lfw_dark --enhanced-root enhanced --stages all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biometric_common as bc


def parse_args():
    p = argparse.ArgumentParser(description="LFW controlled low-light biometric evaluation")
    p.add_argument("--lfw", required=True,
                   help="LFW aligned root (identity subfolders)")
    p.add_argument("--pairs-file", default=None,
                   help="official View-2 pairs.txt or a fold subset "
                        "(default: <lfw>/../pairs.txt)")
    p.add_argument("--dark-root", default="lfw_dark",
                   help="where synthetic low-light LFW images are written")
    p.add_argument("--enhanced-root", required=True,
                   help="root containing per-method subdirectories")
    p.add_argument("--methods", nargs="+", default=None,
                   help="method names; 'input' = identity mapping; default = "
                        "subdirectory names under enhanced-root")
    p.add_argument("--pgdb-root", default=None,
                   help="cloned PGDB-GAN repo for auto-enhance")
    p.add_argument("--pgdb-weights", default=None,
                   help="PGDB-GAN checkpoint for auto-enhance")
    p.add_argument("--gallery-per-id", type=int, default=1,
                   help="normal-light gallery images per identity")
    p.add_argument("--embed-cache", default=None, help="optional .npz embedding cache")
    p.add_argument("--stages", nargs="+",
                   default=["prepare", "enhance", "evaluate"],
                   help="which stages to run (default: all)")
    p.add_argument("--degrade-all", action="store_true",
                   help="degrade/enhance every LFW image instead of only the "
                        "images referenced by --pairs-file")
    bc.add_common_args(p)
    return p.parse_args()


def lfw_identity_dirs(lfw: Path):
    return sorted(d for d in Path(lfw).iterdir() if d.is_dir())


def parse_pairs(pairs_file: Path):
    out = []
    for line in Path(pairs_file).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            name, n1, n2 = parts
            out.append((name, int(n1), name, int(n2), True))
        elif len(parts) == 4:
            name1, n1, name2, n2 = parts
            out.append((name1, int(n1), name2, int(n2), False))
    return out


def lfw_image(lfw: Path, name: str, num: int) -> Path:
    return Path(lfw) / name / f"{name}_{num:04d}.jpg"


def _pairs_needed_set(args, lfw: Path):
    """Relative paths of all images referenced by the pairs file, or None."""
    pairs_file = Path(args.pairs_file) if args.pairs_file else (lfw.parent / "pairs.txt")
    if args.degrade_all or not pairs_file.exists():
        return None, pairs_file
    needed = set()
    for n1, i1, n2, i2, _same in parse_pairs(pairs_file):
        needed.add(lfw_image(lfw, n1, i1).relative_to(lfw).as_posix())
        needed.add(lfw_image(lfw, n2, i2).relative_to(lfw).as_posix())
    return needed, pairs_file


def stage_prepare(args):
    lfw = Path(args.lfw)
    dark_root = Path(args.dark_root)
    params = bc.DEGRADE_PARAMS
    needed, pairs_file = _pairs_needed_set(args, lfw)
    if needed is None:
        print("NOTE: no pairs file found or --degrade-all set; degrading the "
              "entire LFW set (13,233 images)")
    else:
        print(f"restricting degradation to {len(needed)} pair-referenced images "
              f"({pairs_file})")
    alphas, total = {}, 0
    for ident in lfw_identity_dirs(lfw):
        for img in bc.list_images(ident):
            rel = img.relative_to(lfw)
            if needed is not None and rel.as_posix() not in needed:
                continue
            rng = bc.per_image_rng(params["seed"], rel.as_posix())
            alpha = float(rng.uniform(params["alpha_min"], params["alpha_max"]))
            out = dark_root / rel
            if not out.exists():
                x = cv2.imread(str(img))
                if x is not None:
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                    bc.save_image(out, bc.degrade_image(x, alpha, rng, params))
            alphas[rel.as_posix()] = alpha
            total += 1
    out_dir = Path(args.out)
    bc.write_run_config(out_dir / "biometric_lfw_degrade_config.json", {
        "stage": "prepare", "params": params, "n_images": total,
        "restricted_to_pairs": needed is not None,
        "n_needed": len(needed) if needed is not None else total,
        "alphas": alphas, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"prepared {total} low-light images in {dark_root}")


def _enhance_pgdb(args, dark_root: Path, pgdb_dir: Path, files: list):
    """Run the released PGDB-GAN test.py on a flattened copy of the nested
    LFW layout and map its outputs back to the mirrored structure."""
    flat_in = Path(args.out) / "_flat_lfw_dark"
    flat_out = Path(args.out) / "_flat_pgdb_out"
    flat_in.mkdir(parents=True, exist_ok=True)
    flat_out.mkdir(parents=True, exist_ok=True)
    rel_by_stem = {}
    for f in files:
        rel_by_stem[f.stem] = f.relative_to(dark_root)
        dst = flat_in / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    test_py = Path(args.pgdb_root) / "scripts" / "test.py"
    if not test_py.exists():
        test_py = Path(args.pgdb_root) / "test.py"
    if not test_py.exists():
        print(f"test.py not found under {args.pgdb_root}")
        return
    cmd = ["python", str(test_py),
           "--data_path_test_low", str(flat_in),
           "--model_test", str(Path(args.pgdb_weights)),
           "--save", str(flat_out), "--seed", "2", "--gpu", "0"]
    print("running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(Path(args.pgdb_root)))
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"PGDB-GAN auto-run failed: {exc}")
        print(f"Run the released test.py manually and place outputs under "
              f"{pgdb_dir} with the same relative paths as the LFW images.")
        return
    copied = 0
    for out in bc.list_images(flat_out):
        stem = out.stem
        key = stem[:-len("_enhance")] if stem.endswith("_enhance") else stem
        rel = rel_by_stem.get(key)
        if rel is None:
            continue
        dst = pgdb_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(out, dst)
        copied += 1
    print(f"PGDB-GAN outputs mapped: {copied}")


def stage_enhance(args):
    dark_root = Path(args.dark_root)
    root = Path(args.enhanced_root)
    lfw = Path(args.lfw)
    needed, _pairs_file = _pairs_needed_set(args, lfw)
    idents = lfw_identity_dirs(dark_root)
    files = []
    for d in idents:
        for f in bc.list_images(d):
            rel = f.relative_to(dark_root).as_posix()
            if needed is None or rel in needed:
                files.append(f)
    for f in files:
        rel = f.relative_to(dark_root)
        dst = root / "input" / rel
        if not dst.exists():
            x = cv2.imread(str(dark_root / rel))
            if x is not None:
                bc.save_image(dst, cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    pgdb_dir = root / "PGDB-GAN"
    if not any(pgdb_dir.glob("*.*")):
        if args.pgdb_root and args.pgdb_weights:
            _enhance_pgdb(args, dark_root, pgdb_dir, files)
        else:
            print(f"NOTE: PGDB-GAN enhanced images not found; provide them under "
                  f"{pgdb_dir} or pass --pgdb-root/--pgdb-weights.")
    methods = args.methods or sorted(d.name for d in root.iterdir() if d.is_dir())
    print("enhance stage done (baseline methods expected under "
          + str(root) + "/<method>/...)")
    return methods


def stage_evaluate(args):
    lfw = Path(args.lfw)
    root = Path(args.enhanced_root)
    methods = args.methods or sorted(d.name for d in root.iterdir() if d.is_dir())
    pairs_file = Path(args.pairs_file) if args.pairs_file else (lfw.parent / "pairs.txt")
    if not pairs_file.exists():
        raise SystemExit(f"pairs file not found: {pairs_file}")
    pairs = parse_pairs(pairs_file)
    print(f"pairs loaded: {len(pairs)} (full View-2 = 6000)")

    emb = bc.Embedder(rec_name=args.rec_name, root=args.insightface_root,
                      device=args.device, use_mtcnn=args.use_mtcnn)
    cache = bc.load_embed_cache(args.embed_cache) if args.embed_cache else {}
    cache_path = Path(args.embed_cache) if args.embed_cache else None

    from insightface.utils import face_align

    def _rec_embed(vec):
        if vec is None:
            return None
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else None

    ARC_DST = np.float32([[38.2946, 51.6963], [73.5318, 51.5014],
                          [56.0252, 71.7366], [41.5493, 92.3655],
                          [70.7299, 92.2041]])

    def embed_aligned(img_bgr, kps):
        rec = emb.app.models.get("recognition")
        if rec is None or kps is None or img_bgr is None:
            return None
        kps = np.asarray(kps, dtype=np.float64).reshape(5, 2)
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32),
                                           ARC_DST, method=cv2.LMEDS)
        if M is None:
            return None
        size = int(rec.input_size[0])
        aimg = cv2.warpAffine(img_bgr, M, (size, size), borderValue=0)
        return _rec_embed(rec.get_feat(aimg))

    def detect_on_bright(img_bgr):
        if img_bgr is None:
            return None
        if emb.mtcnn is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            try:
                boxes, _probs, points = emb.mtcnn.detect(rgb, landmarks=True)
            except Exception:
                boxes, points = None, None
            if boxes is None or len(boxes) == 0:
                return None
            best = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
            return points[best]
        faces = emb.app.get(img_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]))
        return face.kps

    MISSING = object()
    kps_cache = {}
    for k, v in cache.items():
        if k.startswith("K|") and v is not None:
            kps_cache[k[2:]] = np.asarray(v)

    def precompute_kps(rel_paths):
        todo = list(dict.fromkeys(r for r in rel_paths if r not in kps_cache))
        batch = 32
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            rgbs, paths = [], []
            for rel in chunk:
                img = cv2.imread(str(lfw / rel))
                if img is None:
                    kps_cache[rel] = None
                    continue
                rgbs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                paths.append(rel)
            if not rgbs:
                continue
            try:
                boxes, _probs, points = emb.mtcnn.detect(
                    np.stack(rgbs), landmarks=True)
            except Exception:
                boxes, points = None, None
            if boxes is None:
                boxes = [None] * len(paths)
                points = [None] * len(paths)
            for rel, bs, pts in zip(paths, boxes, points):
                if bs is None or len(bs) == 0 or pts is None:
                    kps_cache[rel] = None
                    continue
                try:
                    best = int(np.argmax(
                        [(b[2] - b[0]) * (b[3] - b[1]) for b in bs]))
                    kps_cache[rel] = np.asarray(pts[best], dtype=np.float64)
                except Exception:
                    kps_cache[rel] = None
            print(f"kps batch {s // batch + 1}/{(len(todo) + batch - 1) // batch} "
                  f"done ({min(s + batch, len(todo))}/{len(todo)})", flush=True)
        for rel in todo:
            v = kps_cache.get(rel)
            cache["K|" + rel] = (None if v is None
                                 else np.asarray(v, dtype=np.float64))
            counter["n"] += 1
            maybe_save()

    counter = {"n": 0}

    def maybe_save():
        if cache_path and counter["n"] >= 200:
            bc.save_embed_cache(cache_path, cache)
            counter["n"] = 0
            print("cache saved", flush=True)

    def embed_dark_via_ref(bright_path, dark_path):
        key = "D|" + str(dark_path)
        if key in cache:
            return cache[key]
        rel = bright_path.relative_to(lfw).as_posix()
        kps = kps_cache.get(rel, MISSING)
        if kps is MISSING:
            bright = cv2.imread(str(bright_path))
            kps = detect_on_bright(bright)
            kps_cache[rel] = kps
        if kps is None:
            cache[key] = None
            counter["n"] += 1
            maybe_save()
            return None
        dark = cv2.imread(str(dark_path))
        e = embed_aligned(dark, kps)
        cache[key] = e
        counter["n"] += 1
        maybe_save()
        return e

    def embed_bright(img_path):
        key = "B|" + str(img_path)
        if key in cache:
            return cache[key]
        img = cv2.imread(str(img_path))
        if img is None:
            cache[key] = None
            counter["n"] += 1
            maybe_save()
            return None
        rel = img_path.relative_to(lfw).as_posix()
        kps = kps_cache.get(rel, MISSING)
        if kps is MISSING:
            kps = detect_on_bright(img)
            kps_cache[rel] = kps
        if kps is None:
            cache[key] = None
            counter["n"] += 1
            maybe_save()
            return None
        e = embed_aligned(img, kps)
        cache[key] = e
        counter["n"] += 1
        maybe_save()
        return e

    pair_rel = set()
    for n1, i1, n2, i2, _same in pairs:
        pair_rel.add(lfw_image(lfw, n1, i1).relative_to(lfw).as_posix())
        pair_rel.add(lfw_image(lfw, n2, i2).relative_to(lfw).as_posix())
    needed_idents = {rel.split("/")[0] for rel in pair_rel}
    print(f"fold identities: {len(needed_idents)}", flush=True)
    needed_rels = set(pair_rel)
    for ident in lfw_identity_dirs(lfw):
        if ident.name not in needed_idents:
            continue
        imgs = bc.list_images(ident)
        if imgs:
            needed_rels.add(imgs[0].relative_to(lfw).as_posix())
    print(f"bright images to detect: {len(needed_rels)}", flush=True)
    precompute_kps(sorted(needed_rels))

    gallery_embs, gallery_ids, gallery_files, id_to_idx = [], [], [], {}
    for ident in lfw_identity_dirs(lfw):
        if ident.name not in needed_idents:
            continue
        imgs = bc.list_images(ident)
        if not imgs:
            continue
        idx = len(id_to_idx)
        id_to_idx[ident.name] = idx
        for img in imgs[:args.gallery_per_id]:
            e = embed_bright(img)
            if e is not None:
                gallery_embs.append(e)
                gallery_files.append(str(img))
                gallery_ids.append(idx)
    if not gallery_embs:
        raise SystemExit("no gallery embeddings; check --lfw and detection")
    gallery_embs = np.asarray(gallery_embs)
    gallery_ids = np.asarray(gallery_ids, dtype=np.int64)
    print(f"gallery: {len(gallery_embs)} images, {len(id_to_idx)} identities", flush=True)

    pair_rows = [(lfw_image(lfw, n1, i1), lfw_image(lfw, n2, i2), same)
                 for n1, i1, n2, i2, same in pairs]

    rec_rows, ver_rows = [], []
    for m in methods:
        mdir = root / m
        if not mdir.is_dir():
            print(f"SKIP {m}: enhanced directory missing ({mdir})")
            continue
        by_rel = {}
        for f in mdir.rglob("*"):
            if f.is_file() and f.suffix.lower() in bc.IM_EXTS:
                by_rel[f.relative_to(mdir).as_posix()] = f

        def dark_counterpart(normal_img: Path):
            return by_rel.get(normal_img.relative_to(lfw).as_posix())

        genuine, impostor, missing = [], [], 0
        for img_a, img_b, same in pair_rows:
            cand_a = dark_counterpart(img_a)
            if cand_a is None:
                missing += 1
                continue
            e_a = embed_dark_via_ref(img_a, cand_a)
            e_b = embed_bright(img_b)
            if e_a is None or e_b is None:
                missing += 1
                continue
            sim = float(np.dot(e_a, e_b))
            (genuine if same else impostor).append(sim)
        curve = bc.verification_curve(genuine, impostor, far_points=(0.001, 0.01))
        ver_rows.append({"method": m, "genuine_pairs": len(genuine),
                         "impostor_pairs": len(impostor), "missing": missing,
                         "auc": curve["auc"],
                         "tar_at_far_0.001": curve["tar_at_far"]["0.001"],
                         "tar_at_far_0.01": curve["tar_at_far"]["0.01"]})

        probe_rows = []
        for img_a, img_b, same in pair_rows:
            if not same:
                continue
            cand_b = dark_counterpart(img_b)
            if cand_b is None:
                continue
            a_name = img_a.parent.name
            if a_name not in id_to_idx:
                continue
            e_b = embed_dark_via_ref(img_b, cand_b)
            if e_b is not None:
                probe_rows.append((e_b, id_to_idx[a_name], str(img_b)))
        if probe_rows:
            gal = np.asarray(gallery_embs)
            gid = np.asarray(gallery_ids, dtype=np.int64)
            acc = {kk: [] for kk in range(1, 6)}
            for probe, pid, ref_file in probe_rows:
                keep = np.array([f != ref_file for f in gallery_files])
                r = bc.recognition_ranks(gal[keep], gid[keep],
                                         np.asarray([probe]),
                                         np.asarray([pid], dtype=np.int64), k=5)
                for kk in range(1, 6):
                    acc[kk].append(r["rank%d" % kk])
            ranks = {"rank%d" % kk: (float(np.mean(acc[kk])) if acc[kk] else float("nan"))
                     for kk in range(1, 6)}
        else:
            ranks = {"rank%d" % kk: float("nan") for kk in range(1, 6)}
        rec_rows.append({"method": m, "probes": len(probe_rows), **ranks})
        print(f"METHOD {m}: TAR@0.1%={ver_rows[-1]['tar_at_far_0.001']:.4f} "
              f"AUC={ver_rows[-1]['auc']:.4f} "
              f"rank1={rec_rows[-1]['rank1']:.4f} rank5={rec_rows[-1]['rank5']:.4f}")

    if cache_path:
        bc.save_embed_cache(cache_path, cache)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "biometric_verification_lfw.csv", "w", encoding="utf-8") as fh:
        fh.write("method,genuine_pairs,impostor_pairs,missing,auc,"
                 "tar_at_far_0.001,tar_at_far_0.01\n")
        for r in ver_rows:
            fh.write(f"{r['method']},{r['genuine_pairs']},{r['impostor_pairs']},"
                     f"{r['missing']},{r['auc']:.6f},"
                     f"{r['tar_at_far_0.001']:.6f},{r['tar_at_far_0.01']:.6f}\n")
    with open(out_dir / "biometric_recognition_lfw.csv", "w", encoding="utf-8") as fh:
        fh.write("method,probes,rank1,rank2,rank3,rank4,rank5\n")
        for r in rec_rows:
            fh.write(f"{r['method']},{r['probes']},{r['rank1']:.6f},{r['rank2']:.6f},"
                     f"{r['rank3']:.6f},{r['rank4']:.6f},{r['rank5']:.6f}\n")
    bc.write_run_config(out_dir / "biometric_lfw_config.json", {
        "experiment": "lfw_dark_biometric",
        "lfw": str(lfw),
        "pairs_file": str(pairs_file),
        "n_pairs": len(pairs),
        "methods": methods,
        "rec_model": args.rec_name,
        "mtcnn": args.use_mtcnn,
        "gallery_per_id": args.gallery_per_id,
        "gallery_images": int(len(gallery_embs)),
        "identities": int(len(id_to_idx)),
        "degrade_params": bc.DEGRADE_PARAMS,
        "verification_rows": ver_rows,
        "recognition_rows": rec_rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"wrote CSVs to {out_dir}", flush=True)


def main():
    args = parse_args()
    stages = set(args.stages)
    if args.dry_run:
        stages -= {"evaluate"}
        print(f"dry-run stages: {sorted(stages)}")
    if "prepare" in stages:
        stage_prepare(args)
    if "enhance" in stages:
        stage_enhance(args)
    if "evaluate" in stages and not args.dry_run:
        stage_evaluate(args)


if __name__ == "__main__":
    main()