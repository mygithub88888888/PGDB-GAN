#!/usr/bin/env python
"""Generate the file-level split manifests referenced in the paper (Section 4.1).

Writes, one image path per line (relative to the repository root):
  splits/LOL_train.txt, splits/LOL_test.txt          (485 / 15 standard public split)
  splits/FiveK_train.txt, splits/FiveK_test.txt      (4,500 / 500 Expert-C split)
  splits/DarkFace_train.txt, splits/DarkFace_test.txt (subject-disjoint partition)

For the paired benchmarks the low/input side is recorded; the normal-light partner of
each line shares the identical file name under the corresponding high/expert_c folder.
The DarkFace manifests simply record the contents of the subject-disjoint train/ and
test/ folders that must already be in place (no individual appears in both folders).

Usage:
  python scripts/make_splits.py --data_root ./data
"""
import argparse
import os

PAIRED = [
    ("LOL_train.txt", "data/LOL/train/low"),
    ("LOL_test.txt", "data/LOL/test/low"),
    ("FiveK_train.txt", "data/FiveK/train/input"),
    ("FiveK_test.txt", "data/FiveK/test/input"),
]
UNPAIRED = [
    ("DarkFace_train.txt", "data/DarkFace/train/image"),
    ("DarkFace_test.txt", "data/DarkFace/test/image"),
]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="./data", help="root of the data/ tree")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    os.makedirs("splits", exist_ok=True)

    for out_name, rel_dir in PAIRED + UNPAIRED:
        d = os.path.join(args.data_root, rel_dir)
        names = []
        if os.path.isdir(d):
            names = sorted(n for n in os.listdir(d) if os.path.splitext(n)[1].lower() in EXTS)
        lines = [os.path.join(rel_dir, n).replace(os.sep, "/") for n in names]
        out_path = os.path.join("splits", out_name)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"{out_path}: {len(lines)} images")


if __name__ == "__main__":
    main()
