# Split manifests

The six file-level manifests referenced in the paper (Section 4.1) are generated with

    python scripts/make_splits.py --data_root ./data

which records one image path per line (relative to the repository root):

| File                      | Dataset              | Expected count |
|---------------------------|----------------------|----------------|
| `LOL_train.txt`           | LOL                  | 485            |
| `LOL_test.txt`            | LOL                  | 15             |
| `FiveK_train.txt`         | MIT-Adobe FiveK      | 4,500          |
| `FiveK_test.txt`          | MIT-Adobe FiveK      | 500            |
| `DarkFace_train.txt`      | DarkFace             | subject-disjoint train set (6,000 images, shipped in this repository) |
| `DarkFace_test.txt`       | DarkFace             | subject-disjoint test set (415 images, shipped in this repository)  |

For the paired benchmarks, each line points to the low-light / RAW input image; its
normal-light partner shares the identical file name under the corresponding
`high` / `expert_c` folder. `LOL_train.txt`, `LOL_test.txt`, `FiveK_train.txt` and
`FiveK_test.txt` are the standard public splits (485/15 and 4,500/500) and are
generated with `make_splits.py` from the official dataset downloads; the two
DarkFace manifests are shipped in this repository. The DarkFace partition is
subject-disjoint by construction: no individual's facial data appears in both
`DarkFace_train.txt` and `DarkFace_test.txt`.
