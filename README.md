# MAC-CDZS Reproduction (Any-Dataset)

An independent, corrected, and generalized reproduction of **MAC-CDZS**
("Contrastive MLP Network Based on Adjacent Coordinates for Cross-Domain
Zero-Shot Hyperspectral Image Classification," IEEE TCSVT 2025), built on
top of the official implementation at
[`jojolee6513/MAC-ZSDA`](https://github.com/jojolee6513/MAC-ZSDA).

This repo fixes the bugs that prevent the original code from running on a
modern environment, and replaces the original's hardcoded per-dataset
`if/elif` chains with a single config file (`dataset_config.py`) — adding
a new hyperspectral dataset, including ones never tested in the original
paper, requires editing **one file**, never the training scripts.

## Quickstart

```bash
pip install -r requirements.txt
```

1. Place your data files (and `Chikusei_imdb_128.pickle`, the fixed source
   domain) wherever you like, and update `DRIVE_ROOT` at the top of
   `dataset_config.py` to point there.
2. The paper's 4 datasets (Indian Pines, Salinas, Pavia University,
   Houston13) are already configured. To use a different dataset, add a
   new entry to `DATASETS` in `dataset_config.py` (template included at
   the bottom of that file).
3. In `train_stage1.py` and `train_stage2.py`, set:
   ```python
   TARGET_NAME = 'IP'   # any key from dataset_config.DATASETS
   SEED_INDEX = 0        # 0-9, picks which of the 10 fixed seeds to use
   ```
4. Run:
   ```bash
   python train_stage1.py
   python train_stage2.py
   ```

That's the entire workflow, for any dataset.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── dataset_config.py        # <-- add new datasets here, nothing else
├── train_stage1.py           # backbone pretraining
├── train_stage2.py           # graph fine-tuning + evaluation
├── self_utils.py
├── aug.py
├── model/
│   ├── GCC_ZSDA.py
│   ├── networks.py
│   └── utils/
│       ├── __init__.py
│       ├── infonce.py
│       └── knn_monitor.py
├── results/                  # your run outputs go here (gitignored except this README note)
└── docs/
    └── walkthrough.md        # detailed bug-by-bug technical writeup (optional, add your own)
```

## Bugs found and fixed (vs. the original repo)

| # | Issue | Fix |
|---|---|---|
| 1 | Dead/incompatible imports (`torch_clustering`, `sklearn_extra`) | Removed |
| 2 | Python 2 iterator syntax (`.next()`) | `next(iter(...))` |
| 3 | Deprecated NumPy aliases | Updated to current equivalents |
| 4 | GPU hardcoded to device 1 (two locations) | Changed to device 0 |
| 5 | `random.sample()` on a `set` | Wrapped in `list(...)` |
| 6 | Prototype matrix initialized with wrong dimension | Fixed to GraphSAGE's actual 64-dim output |
| 7 | Two loss terms computed but never added to the total loss | Restored full loss matching paper's Eq. 12 |
| 8 | **Divide-by-zero** in GraphSAGE neighbor aggregation for nodes with no spatial neighbors — silently froze training from episode 0 | Added `.clamp(min=1)` |
| 9 | Houston13 loader missing required key names | Handled generically via `dataset_config.py`'s `data_key`/`label_key` fields |
| 10 | Visualization bug: wrong variable name left the few-shot classification map permanently black | Corrected variable reference |
| 11 | **GraphSAGE memory blowup**: a dense `(nodes × neighbors)` mask matrix scaled quadratically, causing RAM-exhaustion crashes on certain random seeds | Replaced with sparse `index_add_`-based aggregation, scaling with actual edge count |
| 12 | Stage 2 scripts unconditionally loaded the full Chikusei source dataset even though Stage 2 never uses it — wasted several GB of RAM for the entire run, the root cause of #11's crashes on specific seeds | Removed the unused load entirely from `train_stage2.py` |
| 13 | Each dataset's class counts, paths, and spectral band count were hardcoded directly into the training scripts and model file | Replaced with `dataset_config.py` registry; `GCC_ZSDA.py`'s `ZSDAModel` now accepts `channel` as a constructor argument |
| 14 | A `nn.Embedding(Embedding, 128)` layer took a hardcoded, dataset-specific size that had to be supplied per dataset | Confirmed dead code (every reference lives inside a fully commented-out method) and removed the dependency entirely |

## Known limitation: per-class collapse on small-sample classes

During multi-seed evaluation on Indian Pines, certain random seeds
produced 0% accuracy on specific zero-shot classes. This was diagnosed
(not assumed) via:
- A confusion-matrix trace showing a bimodal pattern (some classes near
  100%, others exactly 0%, nothing in between)
- A negative silhouette score (-0.09) confirming the affected classes'
  features genuinely overlap in feature space for that random split
- Testing three different clustering algorithms (K-means, K-means++,
  K-medoids) — all hit the same wall on the same classes, ruling out a
  clustering-algorithm bug

Root cause: some classes have as few as 2 samples once randomly assigned
to the zero-shot split for a given seed — below any algorithm's ability
to learn a meaningful cluster. This mirrors why the paper itself reports
results as a 10-seed mean ± std rather than a single run. **If you see
this on a new dataset, check the per-class sample counts for whichever
classes collapsed before assuming a bug.**

## Reproduction results (Indian Pines, 10 seeds)

| Metric | This reproduction | Paper |
|---|---|---|
| OA | 55.48 ± 3.40 | 59.72 ± 3.54 |

The standard deviations closely match the paper's, indicating similar
seed-to-seed variance characteristics. The mean gap is most likely
explained by a clustering-algorithm discrepancy: this repo uses K-means;
the paper used CLARA, which requires `scikit-learn-extra`, incompatible
with current NumPy. Confirmed: switching to K-means++ or K-medoids made
results worse, not better, ruling out the clustering algorithm as the
sole explanation and pointing toward upstream feature-quality differences
not yet fully isolated.

## Caveats — read before assuming everything here is verified

- `model/networks.py`, `model/utils/infonce.py`, and
  `model/utils/knn_monitor.py` are used as-is from the original repo and
  were not individually audited line-by-line for bugs — no issues were
  ever traced back to them during this reproduction, but that's not the
  same as a full audit.
- `model/utils/__init__.py` is included as an **empty placeholder file**.
  The original repo's actual `__init__.py` content was never directly
  verified against this reproduction's working runs — if importing
  `model.utils` fails, check what your original copy of this file
  actually contains and restore it.
- Per-class accuracy figures are not directly comparable to the paper's
  own per-class table, since which physical classes get assigned to the
  zero-shot group is randomly determined by the seed.
- Only Indian Pines (10 seeds) has been fully run and verified end-to-end
  in this reproduction. Salinas, Pavia University, and Houston13 are
  configured in `dataset_config.py` but have not yet been run.

## Citation

If referencing the original method, cite:
```
J. Li et al., "Contrastive MLP Network Based on Adjacent Coordinates for
Cross-Domain Zero-Shot Hyperspectral Image Classification," IEEE TCSVT, 2025.
```
