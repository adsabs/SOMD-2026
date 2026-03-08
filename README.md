# SOMD-2026
Work by Felix Grezes and Atilla Alkan on [SOMD 2026](https://nfdi4ds.github.io/nslp2026/docs/somd_shared_task.html)


# SOMD 2026 — Inference Time Measurement

This notebook benchmarks the inference time of the two systems submitted to the **SOMD 2026 Shared Task** (cross-document software mention coreference resolution):

- **FM** — Fuzzy Matching baseline
- **CAR** — Context Aware Representations (embedding-based system using `all-MiniLM-L6-v2`)

Both systems are evaluated on the **Subtask 1** test set (743 mentions, 244 documents) and compared in terms of raw inference time, CoNLL F1, and efficiency (F1/sec).

---

## Repository Structure

```
.
├── inference_time.ipynb        ← Main benchmark notebook
├── data/
│   └── SOMD 2026/
│       └── subtask 1/
│           └── test_data.jsonl ← Test set (mentions with context)
└── predictions/
    ├── fuzzy_matching/
    │   └── pred_clusters_subtask 1.json
    └── car/
        └── pred_clusters_subtask 1.json
```

---

## Requirements

```bash
pip install numpy scikit-learn sentence-transformers torch psutil tqdm
```

> **Note:** A GPU is not required. All timing results reported in the paper were obtained on CPU only (14-core x86_64, 33 GB RAM).

---

## How to Use

### 1. Set the subtask

At the top of **Section 1 (Load Data)**, set the `SUBTASK` variable:

```python
SUBTASK = "subtask 1"   # or "subtask 2" for Subtasks 2 & 3
```

This controls which test file is loaded and where predictions are saved.

### 2. Adjust hyperparameters (according to the subtask)

- **FM threshold** — in Section 3: `THETA = 0.83`
- **CAR distance threshold** — in Section 4: `DELTA = 0.4`
- **CAR embedding weights** — `mention_weight=0.6`, `document_weight=0.4`

These match the tuned values reported in the system description paper.

### 3. Run the notebook end-to-end

Execute all cells in order. Each section is self-contained:

| Section | Description |
|---------|-------------|
| **0. Imports & Setup** | Loads libraries and prints hardware info |
| **1. Load Data** | Reads the test set from `TEST_DATA_PATH` |
| **2. Timing Utility** | Defines `time_system()`: runs a warmup pass, then 5 timed runs |
| **3. Time FM** | Times the Fuzzy Matching system and saves predicted clusters |
| **4. Time CAR** | Times the Context Aware Representations system and saves predicted clusters |
| **5. Summary** | Prints a precision–speed tradeoff table and computes the FM→CAR speedup ratio |

### 4. Set CoNLL F1 scores

In **Section 5**, fill in the official evaluation scores before running:

```python
FM_CONLL_F1  = 0.95   # replace with actual score
CAR_CONLL_F1 = 0.96   # replace with actual score
```

---

## Example Results (Subtask 1 — CPU only)

| System | Mean Time (s) | CoNLL F1 | F1/sec |
|--------|--------------|----------|--------|
| Fuzzy Matching (FM) | 0.354 ± 0.003 | 0.950 | 2.68 |
| Context Aware Repr. (CAR) | 4.454 ± 0.354 | 0.960 | 0.22 |


---

## Methodology

### Timing Protocol

- A **warmup run** is executed before measurement to avoid cold-start effects (e.g. model loading, JIT compilation).
- **5 timed runs** are performed using `time.perf_counter()` for high-resolution wall-clock measurement.
- Results are reported as mean ± standard deviation, with min and max.

### Fuzzy Matching (FM)

A two-stage pipeline:
1. **Exact matching** on normalized mention strings (lowercased, version numbers removed, punctuation stripped).
2. **Fuzzy merging** of clusters using `SequenceMatcher` with threshold `θ = 0.83`.

### Context Aware Representations (CAR)

An embedding-based approach:
1. Mention text and document context (up to 10 sentences) are encoded **separately** using `all-MiniLM-L6-v2`.
2. Both embeddings are L2-normalized and combined as a weighted sum (mention: 0.6, document: 0.4).
3. Agglomerative clustering (cosine distance, average linkage, threshold `δ = 0.4`) produces final clusters.

---

## Citation

If you use this code, please cite our system description paper:

```bibtex
@inproceedings{alkan2026somd,
  title     = {Fine-tuning-free Approaches for Cross-document Software Mention Coreference Resolution},
  author    = {Alkan, Atilla and Grezes, Felix and Bartlett, Jennifer and Kelbert, Anna and Lockhart, Kelly and Accomazzi, Alberto},
  booktitle = {LREC-NSLP SOMD 2026 Shared Task},
  year      = {2026}
}
```
