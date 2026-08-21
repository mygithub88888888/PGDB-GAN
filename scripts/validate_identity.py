#!/usr/bin/env python
"""Identity preservation, face verification, and closed-set recognition.

Implements the exact protocol of the paper (Section 4.5):
  * MTCNN face detection (facenet-pytorch), faces that cannot be localized are excluded.
  * ArcFace ResNet-50 (official insightface w600k_r50, 512-dimensional embeddings),
    cosine similarity.
  * ID-SIM: cosine similarity between the embedding of the enhanced output and that
    of the reference face. The reference is the normal-light ground truth on the
    paired benchmarks (LOL, MIT-Adobe FiveK) and the original low-light input on
    the unpaired DarkFace benchmark (self-reference for the input row, ID-SIM=1.000).
  * Verification (paired benchmarks): positive pairs = each enhanced image and its
    reference; negative pairs = all cross-image combinations. Reports TAR@FAR=0.1%
    (and EER) over the cosine scores.
  * Recognition (paired benchmarks): closed-set protocol; gallery = reference
    embeddings, probes = enhanced outputs; reports rank-1 identification accuracy.

Dependencies: pip install facenet-pytorch insightface onnxruntime

Example:
  python scripts/validate_identity.py --enhanced_dir results/fivek --reference_dir data/FiveK/test/expert_c \
      --ref_kind groundtruth --dataset fivek --gpu 0 --out results/fivek_identity.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_images(root, verbose=False):
    """Return {stem: path} for all images under root (non-recursive first, then recursive)."""
    found = {}
    for base, _dirs, names in os.walk(root):
        for name in sorted(names):
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS:
                found.setdefault(stem, os.path.join(base, name))
    if verbose:
        print(f"  found {len(found)} images under {root}", file=sys.stderr)
    return found


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enhanced_dir", required=True, help="directory of enhanced outputs")
    ap.add_argument("--reference_dir", required=True,
                    help="directory of reference faces (ground truth for paired benchmarks, "
                         "original low-light inputs for DarkFace)")
    ap.add_argument("--ref_kind", choices=["groundtruth", "input"], default="groundtruth",
                    help="semantics of the reference directory")
    ap.add_argument("--dataset", choices=["lol", "fivek", "darkface"], default="fivek")
    ap.add_argument("--gpu", type=int, default=0, help="GPU device id (negative = CPU)")
    ap.add_argument("--out", default=None, help="path of the JSON result file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch
    from facenet_pytorch import MTCNN
    from insightface.app import FaceAnalysis

    device = "cuda" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40,
                  thresholds=[0.6, 0.7, 0.7], post_process=False)

    face = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"])
    face.prepare(ctx_id=args.gpu if device == "cuda" else -1, det_size=(640, 640))

    enhanced = load_images(args.enhanced_dir, args.verbose)
    reference = load_images(args.reference_dir, args.verbose)

    # Pair enhanced and reference images by file stem.
    paired = [(s, enhanced[s], reference[s]) for s in sorted(enhanced) if s in reference]
    if args.verbose:
        print(f"paired {len(paired)} of {len(enhanced)} enhanced images", file=sys.stderr)

    def embed(path):
        """MTCNN localization followed by ArcFace embedding; None if no face found."""
        img = Image.open(path).convert("RGB")
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        x1, y1, x2, y2 = boxes[0]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img.width, int(x2)), min(img.height, int(y2))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        crop = np.asarray(img.crop((x1, y1, x2, y2)))
        rec = face.get(crop)
        if not rec:
            return None
        return np.asarray(rec[0].normed_embedding, dtype=np.float32)

    sims = []
    skipped = 0
    emb_enh, emb_ref, pair_keys = [], [], []
    for stem, enh_path, ref_path in paired:
        e = embed(enh_path)
        r = embed(ref_path)
        if e is None or r is None:
            skipped += 1
            if args.verbose:
                print(f"  skipped {stem}: face not localized", file=sys.stderr)
            continue
        pair_keys.append(stem)
        emb_enh.append(e)
        emb_ref.append(r)
        sims.append(cosine(e, r))

    sims = np.asarray(sims, dtype=np.float64)
    id_sim = float(sims.mean()) if sims.size else float("nan")

    results = {
        "dataset": args.dataset,
        "ref_kind": args.ref_kind,
        "paired_images": len(paired),
        "evaluated_images": int(sims.size),
        "skipped_no_face": skipped,
        "id_sim_mean": id_sim,
        "id_sim_std": float(sims.std()) if sims.size else float("nan"),
        "id_sim_min": float(sims.min()) if sims.size else float("nan"),
        "id_sim_max": float(sims.max()) if sims.size else float("nan"),
    }

    if args.ref_kind == "groundtruth" and len(emb_enh) > 1:
        # Verification: positive = matched pairs; negative = all cross pairs.
        emb_enh = np.asarray(emb_enh, dtype=np.float32)
        emb_ref = np.asarray(emb_ref, dtype=np.float32)
        n = len(emb_enh)
        pos = np.asarray(sims, dtype=np.float64)
        neg = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                neg.append(cosine(emb_enh[i], emb_ref[j]))
        neg = np.asarray(neg, dtype=np.float64)
        # FAR=0.1% threshold over the negative (impostor) score distribution.
        far = 0.001
        thr = np.quantile(neg, 1.0 - far) if neg.size else float("nan")
        tar = float((pos >= thr).mean())
        # EER via simple search.
        pos_sorted = np.sort(pos)
        neg_sorted = np.sort(neg)
        best = 1.0
        for t in np.concatenate([pos_sorted, neg_sorted]):
            far_t = float((neg >= t).mean())
            frr_t = float((pos < t).mean())
            best = min(best, max(far_t, frr_t))
        eer = best
        # Closed-set recognition: gallery = reference embeddings, probes = enhanced.
        gallery = emb_ref
        hits = 0
        for i in range(n):
            d = [cosine(emb_enh[i], gallery[j]) for j in range(n)]
            if int(np.argmax(d)) == i:
                hits += 1
        rank1 = hits / n
        results.update({
            "verification_negative_pairs": int(neg.size),
            "tar_at_far_0.1pct": tar,
            "eer": eer,
            "rank1_closed_set": rank1,
        })

    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
