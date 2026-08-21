#!/usr/bin/env python
"""Unified metric protocols of the paper (Section 4.2, Tables 3 and 5).

Metrics implemented here:
  * NSR  -- Noise Suppression Rate. Per image, the noise standard deviation sigma is
            estimated with the Median Absolute Deviation (MAD) of the finest-scale HH
            wavelet subband coefficients (Daubechies-4 wavelet, sigma = MAD / 0.6745).
            NSR = (sigma_input - sigma_enhanced) / sigma_input * 100 (%). By definition
            the input row has NSR = 0.00%.
  * LPIPS -- official AlexNet-based implementation, lower is better. On paired
            benchmarks it is computed against the normal-light ground truth; on the
            unpaired DarkFace benchmark it is computed against the original low-light
            input (identity drift).
  * FID   -- Frechet Inception Distance, Inception-v3 pool3 features, all images
            resized to 299x299. The reference distribution is the normal-light ground
            truth on paired benchmarks and the 500 Expert-C images of the MIT-Adobe
            FiveK test split on the unpaired DarkFace benchmark.
  * NIQE  -- standard no-reference implementation with default parameters.

Dependencies: pip install torch torchvision torchmetrics piq PyWavelets

Example (paired):
  python scripts/metrics.py --metric nsr --dir1 data/LOL/test/low --dir2 results/lol_enhanced
  python scripts/metrics.py --metric lpips --dir1 results/lol_enhanced --dir2 data/LOL/test/high
  python scripts/metrics.py --metric fid --dir1 results/lol_enhanced --dir2 data/LOL/test/high
  python scripts/metrics.py --metric niqe --dir1 results/lol_enhanced
Example (DarkFace, unpaired):
  python scripts/metrics.py --metric nsr --dir1 data/DarkFace/test/image --dir2 results/darkface_enhanced
  python scripts/metrics.py --metric lpips --dir1 results/darkface_enhanced --dir2 data/DarkFace/test/image
  python scripts/metrics.py --metric fid --dir1 results/darkface_enhanced --dir2 data/FiveK/test/expert_c
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root):
    found = []
    for base, _dirs, names in os.walk(root):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(base, name))
    return found


def nsr(dir_input, dir_enhanced):
    import pywt
    sigmas = []
    for img_in, img_out in zip(list_images(dir_input), list_images(dir_enhanced)):
        a = np.asarray(Image.open(img_in).convert("L"), dtype=np.float64) / 255.0
        b = np.asarray(Image.open(img_out).convert("L"), dtype=np.float64) / 255.0
        cA, (cH, cV, cD) = pywt.dwt2(a, "db4")
        sigma_in = np.median(np.abs(cD - np.median(cD))) / 0.6745
        _, (cH2, cV2, cD2) = pywt.dwt2(b, "db4")
        sigma_out = np.median(np.abs(cD2 - np.median(cD2))) / 0.6745
        sigmas.append((sigma_in - sigma_out) / (sigma_in + 1e-12) * 100.0)
    return np.asarray(sigmas)


def lpips(dir_a, dir_b):
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchvision import transforms
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    to_tensor = transforms.ToTensor()
    values = []
    for fa, fb in zip(list_images(dir_a), list_images(dir_b)):
        ta = to_tensor(Image.open(fa).convert("RGB")).unsqueeze(0)
        tb = to_tensor(Image.open(fb).convert("RGB")).unsqueeze(0)
        values.append(float(metric(ta, tb)))
    return np.asarray(values)


def fid(dir_generated, dir_reference):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms
    metric = FrechetInceptionDistance(feature=2048, normalize=True)
    tf = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    ref = [tf(Image.open(p).convert("RGB")) for p in list_images(dir_reference)]
    gen = [tf(Image.open(p).convert("RGB")) for p in list_images(dir_generated)]
    for im in ref:
        metric.update(im.unsqueeze(0), real=True)
    for im in gen:
        metric.update(im.unsqueeze(0), real=False)
    return float(metric.compute())


def niqe(dir_images):
    from piq import niqe as _niqe
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    values = []
    for p in list_images(dir_images):
        t = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        values.append(float(_niqe(t, data_range=1.0, reduction="none")))
    return np.asarray(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", required=True, choices=["nsr", "lpips", "fid", "niqe"])
    ap.add_argument("--dir1", required=True, help="first directory (input for NSR, generated for FID/LPIPS/NIQE)")
    ap.add_argument("--dir2", help="second directory (enhanced for NSR, reference for LPIPS/FID)")
    args = ap.parse_args()

    if args.metric == "nsr":
        vals = nsr(args.dir1, args.dir2)
    elif args.metric == "lpips":
        vals = lpips(args.dir1, args.dir2)
    elif args.metric == "fid":
        print(fid(args.dir1, args.dir2))
        return
    else:
        vals = niqe(args.dir1)

    print(f"mean={vals.mean():.4f} std={vals.std(ddof=1):.4f} n={vals.size}")
    print(f"min={vals.min():.4f} max={vals.max():.4f}")


if __name__ == "__main__":
    main()
