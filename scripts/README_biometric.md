# Biometric Evaluation Scripts (Experiments 1-3)

Two entry scripts plus one shared module (`biometric_common.py`):

| Script | Experiment | Output (under `--out`) |
| --- | --- | --- |
| `identity_fidelity.py` | (1) Identity-preservation fidelity: ArcFace cosine between original and enhanced image, mean +/- std per method | `biometric_identity_fidelity.csv` |
| `lfw_dark_biometric.py` | (2) Controlled low-light recognition Rank-1/5 and (3) verification TAR@FAR=1%/ROC AUC on LFW | `biometric_recognition_lfw.csv`, `biometric_verification_lfw.csv` |

Every run writes a `*_config.json` with all paths, parameters, model names,
sample counts and missing-face counts, so the experiments are fully
reproducible.

## 1. Install

    pip install insightface onnxruntime opencv-python numpy
    pip install facenet-pytorch     # optional, only needed with --use-mtcnn

The recognition model is pinned to InsightFace `buffalo_l` (SCRFD-10GF
detection + ArcFace-R50 recognition). Because github.com downloads may be
unreachable, pre-download the five weights `det_10g.onnx`, `w600k_r50.onnx`,
`2d106det.onnx`, `1k3d68.onnx`, `genderage.onnx` and place them in
`%USERPROFILE%\.insightface\models\buffalo_l\`.

## Results (released CSVs)

| Method | Identity fidelity (mean cosine) | Rank-1 (%) | Rank-5 (%) | Verification AUC | TAR@FAR=0.1% (%) |
| --- | --- | --- | --- | --- | --- |
| input (no enhancement) | 1.000 | 42.33 | 58.33 | 0.9288 | 49.00 |
| SCI | 0.656 | 92.67 | 95.67 | 0.9851 | 93.33 |
| SNR-Net | 0.603 | 87.67 | 92.58 | 0.9631 | 89.29 |
| Retinexformer | 0.584 | 91.33 | 95.67 | 0.9850 | 90.00 |
| PGDB-GAN (ours) | 0.543 | 90.67 | 94.67 | 0.9850 | 92.67 |

SNR-Net values were produced with the official `LOLv1.pth` checkpoint.

## 2. Experiment 1 - DarkFace identity fidelity

Layout:

    enhanced_root/
      SCI/          <same stem>.png  ...  (stem must equal the original)
      SNR-Net/      ...
      Retinexformer/ ...
      PGDB-GAN/     ...                   (or auto-run via --pgdb-root/--pgdb-weights)

The `input` row is 1.0 by construction (identity mapping) and needs no
files. Run:

    python identity_fidelity.py --originals <darkface_low_light_dir> `
      --enhanced-root <enhanced_root> `
      --methods input SCI SNR-Net Retinexformer PGDB-GAN `
      --labels-dir <optional_bbox_txt_dir> `
      --out results/biometric_results

`--labels-dir` accepts per-image .txt with a 4-number box line
(`x1 y1 x2 y2`) or a WIDER-Face ellipse line (>=5 numbers); a leading
face-count line is ignored. Use `--dry-run` to check counts first and
`--embed-cache cache.npz` to reuse embeddings. If PGDB-GAN auto-run writes
`<stem>_enhance.png` files, they are renamed to plain stems automatically.

Baseline commands (pinned commits/checkpoints, see repo README):

    SCI          : python CVPR/test.py --data_path <low_input_dir> --save_path <out> --model CVPR/weights/medium.pt --gpu 0
    SNR-Net      : python test.py -opt options/test/<dataset>.yml   (set pretrain_model_G to the official checkpoint)
    Retinexformer: python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_<DATASET>.yml --weights pretrained_weights/<DATASET>.pth --dataset <DATASET>

## 3. Experiments 2-3 - LFW controlled low-light biometrics

Required data: LFW aligned images (`<lfw>/<identity>/<identity>_NNNN.jpg`,
13,233 images / 5,749 identities) and a View-2 pairs file (default
location: next to the `lfw` folder).

Two protocol sizes, same script:

- Fast (recommended): official View-2 fold 1, `pairs_fold1.txt` (300
  genuine / 300 impostor). With `--pairs-file` set, `prepare` and
  `enhance` automatically process only the pair-referenced images
  (about 1,100-1,300 images) instead of all 13,233.
- Full (if more evidence is requested): the complete `pairs.txt` (6,000
  pairs) plus `--degrade-all`; nothing else changes.

Step A - synthesize low-light LFW (deterministic, resume-safe):

    python lfw_dark_biometric.py --lfw <lfw_aligned> --pairs-file <pairs_fold1.txt> `
      --dark-root lfw_dark --enhanced-root enhanced --stages prepare

Degradation (in `biometric_common.py:DEGRADE_PARAMS`; CONFIRM WITH THE
AUTHORS BEFORE PUBLISHING):

    y = clip(alpha * x^gamma + N(0, sigma_g^2)
             + N(0, (sigma_p * sqrt(alpha * x^gamma))^2), 0, 1)
    gamma = 2.2, alpha ~ U[0.08, 0.20] per image,
    sigma_g = 0.03, sigma_p = 0.05, seed = 2024

Per-image alpha and noise are hashed from (seed, relative path), so re-runs
and interrupted runs produce identical images.

Step B - enhance the degraded images with every method. Baselines are run
externally with the commands in Section 2 (same official checkpoints and
repo commits); outputs must mirror the LFW structure:

    enhanced/SCI/<identity>/<identity>_NNNN.jpg
    enhanced/PGDB-GAN/<identity>/<identity>_NNNN.jpg

If a baseline loader does not recurse into identity subfolders, flatten
`lfw_dark` into a temporary flat directory (stems are globally unique),
run it, and map the outputs back. The `input` copies are written
automatically; PGDB-GAN auto-run (`--pgdb-root/--pgdb-weights`) already
flattens and remaps automatically.

Step C - evaluate (both statistic blocks in one pass):

    python lfw_dark_biometric.py --lfw <lfw_aligned> --pairs-file <pairs_fold1.txt> `
      --dark-root lfw_dark --enhanced-root enhanced `
      --methods input SCI SNR-Net Retinexformer PGDB-GAN --stages evaluate

Protocol: gallery = normal-light LFW images (one per identity);
verification probe = enhanced image of the first pair member vs the
normal-light second member (TAR@FAR=1%, ROC AUC); recognition probe =
enhanced dark images against the normal-light gallery (Rank-1/5). With 300
impostor pairs, TAR@FAR=0.1% is below resolution (the script prints nan):
report AUC and TAR@FAR=1% and state the sample size in the paper. Use
`--embed-cache cache.npz` so normal-light embeddings are computed once and
shared by all methods.

## 4. Cloud / environment notes

- Python >= 3.8 recommended (insightface wheels); numpy >= 1.20
  (`np.trapezoid` has an `np.trapz` fallback).
- CUDA + onnxruntime-gpu recommended; CPU works with `--device -1`.
- Before running PGDB-GAN `test.py` on the cloud, remove the hardcoded
  local paths in it (e.g. `F:/github-code/...` in `root_dir` and in the
  logging FileHandler), otherwise it crashes on a machine without drive F:.
- Windows console: the commands above use PowerShell line continuation.

## 5. Approximate runtimes (single 16 GB GPU)

- Experiment 1 (415 DarkFace images x 5 methods): baseline inference 15-40
  min total (SCI dominates), ArcFace embedding 5-10 min.
- Experiments 2-3, fold-1 mode (~1,200 images): prepare 1-2 min (CPU),
  SCI 15-25 min, SNR-Net ~5 min, Retinexformer ~5 min, PGDB-GAN 2-5 min,
  ArcFace embedding ~10 min. Total about 40-60 min.
- Experiments 2-3, full 13,233-image mode: 2.5-6 h, dominated by SCI.
  One-time cost, no training.