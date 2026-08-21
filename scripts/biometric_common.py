"""Shared helpers for the downstream biometric evaluation scripts.

Pipeline: image -> face detection -> 112x112 aligned crop -> ArcFace
embedding -> cosine statistics / rank / TAR@FAR.

All randomness is reproducible: degradation parameters are fixed and the
per-image noise is derived deterministically from the master seed and the
image's relative path (see DEGRADE_PARAMS and per_image_rng).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IM_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# >>> CONFIRM THESE PARAMETERS WITH THE AUTHORS BEFORE PUBLISHING. <<<
# gamma   : display gamma used to convert normal-light LFW images to low light.
# alpha   : per-image exposure scale, drawn uniformly in [alpha_min, alpha_max].
# sigma_g : additive Gaussian noise std (pixel range [0,1]).
# sigma_p : signal-dependent noise coefficient, std = sigma_p * sqrt(x).
# seed    : master seed; per-image randomness is a deterministic hash of
#           (seed, relative path), so re-runs and partial resumes are identical.
DEGRADE_PARAMS = {
    "gamma": 2.2,
    "alpha_min": 0.08,
    "alpha_max": 0.20,
    "sigma_g": 0.03,
    "sigma_p": 0.05,
    "seed": 2024,
}


def list_images(dirpath: Path, exts=IM_EXTS):
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        return []
    return sorted(p for p in dirpath.iterdir() if p.suffix.lower() in exts)


def list_images_recursive(dirpath: Path):
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        return []
    return sorted(f for f in dirpath.rglob("*")
                  if f.is_file() and f.suffix.lower() in IM_EXTS)


def per_image_rng(seed, rel_path: str):
    digest = hashlib.sha256("{}:{}".format(seed, rel_path).encode("utf-8")).digest()
    return np.random.RandomState(np.frombuffer(digest[:4], dtype="<u4")[0])


def degrade_image(img_rgb, alpha=None, rng=None, params=DEGRADE_PARAMS):
    """Convert a normal-light RGB image into a low-light observation.

    model: y = alpha * x^gamma + n_gauss + n_shot,
           n_shot ~ N(0, (sigma_p * sqrt(alpha * x^gamma))^2),
    clipped to [0,1].
    """
    x = np.asarray(img_rgb, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    x_g = np.power(x, float(params["gamma"]))
    if alpha is None:
        rng = rng if rng is not None else np.random
        alpha = float(rng.uniform(params["alpha_min"], params["alpha_max"]))
    x_e = x_g * float(alpha)
    if rng is None:
        rng = np.random
    noise_g = rng.normal(0.0, float(params["sigma_g"]), x_e.shape).astype(np.float32)
    noise_p = (float(params["sigma_p"]) * np.sqrt(np.maximum(x_e, 0.0))
               * rng.normal(0.0, 1.0, x_e.shape).astype(np.float32))
    y = x_e + noise_g + noise_p
    return np.clip(y, 0.0, 1.0)


def save_image(path: Path, img01):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(np.asarray(img01, dtype=np.float32), 0.0, 1.0)
    out = (out * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


class Embedder:
    """Face detection + ArcFace embedding via insightface (buffalo_l)."""

    def __init__(self, rec_name="buffalo_l", root="~/.insightface",
                 device=0, use_mtcnn=False):
        self.rec_name = rec_name
        self.use_mtcnn = use_mtcnn
        self.mtcnn = None
        if use_mtcnn:
            try:
                from facenet_pytorch import MTCNN
                self.mtcnn = MTCNN(keep_all=True,
                                   device="cuda" if device >= 0 else "cpu")
            except ImportError:
                print("WARNING: facenet_pytorch not installed; "
                      "falling back to insightface detection.")
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            sys.exit("Please install insightface:  pip install insightface onnxruntime")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device >= 0 else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name=rec_name, root=str(Path(root).expanduser()),
                                providers=providers)
        self.app.prepare(ctx_id=device, det_size=(640, 640))

    def _embed_with_app(self, img_bgr):
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]))
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = face.embedding
        emb = np.asarray(emb, dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-12)

    def embed_path(self, img_path: Path):
        img = cv2.imread(str(img_path))
        return None if img is None else self.embed_bgr(img)

    def embed_bgr(self, img_bgr):
        if self.mtcnn is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(img_rgb)
            if boxes is None or len(boxes) == 0:
                return self._embed_with_app(img_bgr)
            h, w = img_bgr.shape[:2]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                mw, mh = int(0.4 * (x2 - x1)), int(0.4 * (y2 - y1))
                x1, y1 = max(0, x1 - mw), max(0, y1 - mh)
                x2, y2 = min(w, x2 + mw), min(h, y2 + mh)
                canvas[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]
            emb = self._embed_with_app(canvas)
            return emb if emb is not None else self._embed_with_app(img_bgr)
        return self._embed_with_app(img_bgr)

    def embed_crop(self, img_bgr, box, margin=0.4):
        """Embed using a given bounding box (x1, y1, x2, y2), with the
        remaining image area masked out before re-detection."""
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        mw, mh = int(margin * (x2 - x1)), int(margin * (y2 - y1))
        x1, y1 = max(0, x1 - mw), max(0, y1 - mh)
        x2, y2 = min(w, x2 + mw), min(h, y2 + mh)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]
        emb = self._embed_with_app(canvas)
        if emb is not None:
            return emb
        rec = self.app.models.get("recognition")
        if rec is None or x2 <= x1 or y2 <= y1:
            return None
        crop = img_bgr[y1:y2, x1:x2]
        try:
            crop = cv2.resize(crop, tuple(rec.input_size),
                              interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            return None
        vec = rec.get_feat(crop)
        if vec is None:
            return None
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else None


def cosine_matrix(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def recognition_ranks(gallery_embs, gallery_ids, probe_embs, probe_ids, k=5):
    """Closed-set identification: cumulative Rank-1..Rank-k hit rates."""
    sims = cosine_matrix(probe_embs, gallery_embs)
    order = np.argsort(-sims, axis=1)
    hits = np.zeros(k, dtype=np.int64)
    for i in range(len(probe_ids)):
        ranked_ids = gallery_ids[order[i]]
        pos = np.where(ranked_ids == probe_ids[i])[0]
        if len(pos) == 0:
            continue
        first = pos[0]
        for j in range(k):
            if first <= j:
                hits[j:] += 1
                break
    n = len(probe_ids)
    return {"rank{}".format(kk): (float(hits[kk - 1]) / n if n else 0.0)
            for kk in range(1, k + 1)}


def verification_curve(genuine_sims, impostor_sims, far_points=(0.001, 0.01)):
    """TAR@FAR at the requested operating points and ROC AUC."""
    genuine_sims = np.asarray(genuine_sims, dtype=np.float64)
    impostor_sims = np.asarray(impostor_sims, dtype=np.float64)
    n_gen, n_imp = len(genuine_sims), len(impostor_sims)
    if n_gen == 0 or n_imp == 0:
        return {"auc": float("nan"), "tar_at_far": {}}
    thresholds = np.unique(np.concatenate([genuine_sims, impostor_sims]))
    far = np.array([(impostor_sims >= t).sum() / n_imp for t in thresholds])
    tar = np.array([(genuine_sims >= t).sum() / n_gen for t in thresholds])
    order = np.argsort(far)
    far_s, tar_s = far[order], tar[order]
    far_s = np.concatenate([[0.0], far_s, [1.0]])
    tar_s = np.concatenate([[tar_s[0]], tar_s, [tar_s[-1]]])
    try:
        auc = float(np.trapezoid(tar_s, far_s))
    except AttributeError:
        auc = float(np.trapz(tar_s, far_s))
    tar_at_far = {}
    for fp in far_points:
        idx = np.where(far <= fp)[0]
        tar_at_far[str(fp)] = float(tar[idx].max()) if len(idx) else float("nan")
    return {"auc": auc, "tar_at_far": tar_at_far}


def save_embed_cache(cache_path: Path, cache: dict):
    """Persist embeddings to .npz; paths are stored as an object array."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k, v in cache.items() if v is not None]
    vals = [cache[k] for k in keys]
    np.savez(cache_path, keys=np.array(keys, dtype=object),
             **{"arr_{}".format(i): v for i, v in enumerate(vals)})


def load_embed_cache(cache_path: Path) -> dict:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    data = np.load(cache_path, allow_pickle=True)
    if "keys" not in data:
        return {}
    keys = [str(k) for k in data["keys"].tolist()]
    return {keys[i]: data["arr_{}".format(i)] for i in range(len(keys))}


def write_run_config(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def add_common_args(parser):
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device id; -1 = CPU")
    parser.add_argument("--rec-name", default="buffalo_l",
                        help="insightface model name (pinned ArcFace model)")
    parser.add_argument("--use-mtcnn", action="store_true",
                        help="use MTCNN (facenet-pytorch) as detector, matching the paper")
    parser.add_argument("--insightface-root", default="~/.insightface",
                        help="insightface model cache directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate paths and count images without loading models")
    parser.add_argument("--out", default="results/biometric_results",
                        help="output directory for CSVs and run-config JSON")