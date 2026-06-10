# Multi-Label Emotion Classification

A systematic comparative study of transformation-based multi-label classification (MLC) methods on the **Emotions benchmark dataset**, evaluating 16 method–classifier combinations across 10 evaluation metrics.

> **Published Research — Springer ICIoTCT 2025**  
> *"Comparison of Multilabel Classification Techniques for Emotion Recognition"*  
> Vibhavari Tummewar, Sanyam Shukla, Manasi Gyanchandani  
> Department of CSE, MANIT Bhopal  
> *AIoT — Integration of Smart Intelligence and IoT: Proceedings of the 10th International Conference on Internet of Things and Connected Technologies (ICIoTCT 2025), Springer*

---

## Overview

Emotion recognition is inherently multi-label — a person can simultaneously experience joy and surprise, or sadness and fear. Standard single-label classifiers fail to capture this. This study benchmarks four transformation-based MLC strategies paired with four classical classifiers to identify which combinations best balance accuracy, efficiency, and label-dependency modeling.

**Dataset:** Emotions (593 samples, 72 audio features, 6 emotion labels: anger, disgust, fear, joy, sadness, surprise)

---

## Methods Compared

| Strategy | Description |
|----------|-------------|
| **Binary Relevance (BR)** | Decomposes MLC into independent binary problems per label. Simple, fast, ignores label dependencies. |
| **Classifier Chain (CC)** | Each label classifier uses previous label predictions as features. Models sequential dependencies but prone to error propagation. |
| **Label Powerset (LP)** | Treats each unique label combination as a single class. Captures full label interactions; suffers at scale. |
| **RAkEL** | Ensemble over random label subsets. Balances dependency modeling and computational efficiency. |

**Base classifiers:** Naive Bayes · Logistic Regression · Random Forest · Linear SVM

---

## Key Results

### Subset Accuracy & Hamming Loss

| Method | Classifier | Subset Acc ↑ | Hamming Loss ↓ |
|--------|-----------|:------------:|:--------------:|
| Binary Relevance | Random Forest | 0.3277 | **0.1765** |
| Binary Relevance | Logistic Regression | 0.3109 | 0.2073 |
| Binary Relevance | SVM (Linear) | 0.3109 | 0.2031 |
| Binary Relevance | Naive Bayes | 0.2269 | 0.2451 |
| Classifier Chain | SVM (Linear) | 0.3193 | 0.2143 |
| Classifier Chain | Random Forest | 0.3109 | 0.1863 |
| **Label Powerset** | **Random Forest** | **0.4202** | 0.1863 |
| RAkEL | Random Forest | 0.3950 | 0.1176 |

### Micro-F1 & Macro-F1

| Method | Classifier | Micro-F1 ↑ | Macro-F1 ↑ |
|--------|-----------|:----------:|:----------:|
| **Binary Relevance** | **Random Forest** | **0.7433** | 0.6985 |
| RAkEL | Random Forest | 0.6867 | **0.7111** |
| Label Powerset | Random Forest | 0.6737 | 0.7020 |
| RAkEL | SVM (Linear) | 0.6563 | 0.6667 |
| RAkEL | Logistic Regression | 0.6422 | 0.6437 |

### Best Combinations by Metric

| Metric | Best Combination | Score |
|--------|-----------------|-------|
| Subset Accuracy | RF + Label Powerset | **0.4202** |
| Micro-F1 | RF + Binary Relevance | **0.7433** |
| Macro-F1 | RF + RAkEL | **0.7111** |
| Hamming Loss | RF + Binary Relevance | **0.1765** |

> **Practical recommendation:** Random Forest + Binary Relevance gives the best balance of performance and simplicity (Micro-F1: 0.7433, Hamming Loss: 0.1765) with the lowest computational overhead.

---

## Repository Structure

```
Multilabel-Emotion-Classification/
├── experiments.py          # Full 4×4 grid experiment (all methods × all classifiers)
├── requirements.txt
├── results/
│   ├── results.csv         # Raw metrics for all 16 combinations
│   └── figures/
│       ├── f1_heatmap.png          # Micro-F1 across methods & classifiers
│       ├── hamming_heatmap.png     # Hamming Loss heatmap
│       └── rakel_rf_radar.png      # Radar chart: best model (RAkEL + RF)
└── paper/
    └── multilabel_emotion_classification.pdf
```

---

## Setup & Usage

```bash
# Clone the repository
git clone https://github.com/Vi-bha/Multilabel-Emotion-Classification.git
cd multilabel-emotion-classification

# Install dependencies
pip install -r requirements.txt

# Run all experiments (16 combinations, saves results to results/results.csv)
python experiments.py
```

Results are printed to stdout and saved to `results/results.csv`.

---

## Dependencies

```
scikit-multilearn==0.2.0
scikit-learn>=1.0
numpy>=1.21
pandas>=1.3
liac-arff>=2.5
matplotlib>=3.4
seaborn>=0.11
```

Or simply: `pip install -r requirements.txt`

---

## Methodology

```
Emotions Dataset (593 samples, 6 labels)
          │
          ▼
  Text Normalization → Tokenization → TF-IDF Vectorization
          │
          ▼
   80/20 Train-Test Split (random_state=42)
          │
          ▼
  4 Classifiers × 4 Transformation Methods = 16 Combinations
          │
          ▼
  Evaluation: Subset Accuracy, Hamming Loss, Micro/Macro F1,
              Precision, Recall, Jaccard (micro + samples)
```

---

## Findings

- **Random Forest dominates** — consistently best across all transformation methods, regardless of how the MLC problem is formulated. Its ensemble nature handles label correlations inherent in emotional data.
- **Simpler isn't worse** — Binary Relevance with Random Forest outperforms more sophisticated dependency-modeling methods (CC, RAkEL) on Micro-F1, suggesting that for this dataset, increased complexity introduces overfitting without meaningful benefit.
- **Label Powerset peaks on exact match** — achieves the highest subset accuracy (0.4202) by modeling full label combinations, but requires a strong classifier (RF) to handle the resulting class imbalance.
- **Naive Bayes underperforms** — its feature independence assumption is violated by the correlated emotional features in this dataset.
- **Metric choice matters** — the "best" combination changes depending on whether you optimize for exact label-set matches (LP+RF), per-label accuracy (BR+RF), or balanced performance across labels (RAkEL+RF).

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{tummewar2025multilabel,
  title     = {Comparison of Multilabel Classification Techniques for Emotion Recognition},
  author    = {Vibhavari Tummewar and Sanyam Shukla and Manasi Gyanchandani},
  booktitle = {AIoT -- Integration of Smart Intelligence and IoT: Proceedings of the 10th International
               Conference on Internet of Things and Connected Technologies (ICIoTCT 2025)},
  publisher = {Springer},
  year      = {2025}
}
```

---

## Related Work

This repository complements [LASRMIS](https://github.com/<your-username>/lasrmis) — a Scopus-indexed medical AI system — demonstrating breadth across both classical ML benchmarking and production medical AI.
