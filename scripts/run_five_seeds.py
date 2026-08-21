#!/usr/bin/env python
"""Five-seed batch runner.

Re-runs a command once per independent trial seed so that every PGDB-GAN experiment
can be reproduced with the five distinct seeds {2, 7, 42, 123, 2024} declared in the
paper (Section 4.1). A fresh weight initialization, mini-batch ordering, and
augmentation sampling are drawn for each seed; the data split is fixed.

Usage:
  python scripts/run_five_seeds.py \
      --cmd "python scripts/train_stage1.py --batch_size 16 --lr 2e-4 --epochs 3000 --seed {SEED} --gpu 0"
  python scripts/run_five_seeds.py \
      --cmd "python scripts/test.py --model_test ./weights/LOL.pt --seed {SEED} --gpu 0"
"""
import argparse
import datetime
import subprocess
import sys

SEEDS = [2, 7, 42, 123, 2024]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmd", required=True,
                    help="command template; the literal token {SEED} is replaced by each seed")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                    help="comma-separated seed list (default: 2,7,42,123,2024)")
    ap.add_argument("--log_dir", default="./results/five_seeds",
                    help="directory for per-seed stdout/stderr logs")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    import os
    os.makedirs(args.log_dir, exist_ok=True)

    rc = 0
    for seed in seeds:
        cmd = args.cmd.replace("{SEED}", str(seed))
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log = os.path.join(args.log_dir, f"seed_{seed}_{stamp}.log")
        print(f"[seed {seed}] {cmd}")
        with open(log, "w", encoding="utf-8") as f:
            ret = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        print(f"[seed {seed}] exit={ret.returncode} log={log}")
        rc = max(rc, ret.returncode)
    sys.exit(rc)


if __name__ == "__main__":
    main()
