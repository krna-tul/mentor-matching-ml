# Comparing Machine Learning Methods for Mentor–Mentee Competency Matching

Code and data for the study *“Comparing Machine Learning Methods for Matching Mentors’ Teaching Competencies with Mentees’ Professional Development Needs in Higher Education”* (K. Tulegenova, Astana IT University, 2026).

The study compares **logistic regression (LR)**, **random forest (RF)** and **gradient boosting (GB)** for deciding whether a competency (a mentee’s development need) is relevant to a course text (what a mentor teaches). It uses an explanatory sequential mixed-methods design: a statistical model comparison followed by qualitative content analysis of the errors.

## Data

`manual_records.csv` contains 1,098 manually annotated competency–course pairs from the public **UniSkill** dataset (synthetic records excluded):

* N. Musazade, J. Mezei, and M. Zhang, “UniSkill: A dataset for matching university curricula to professional competencies,” *LREC 2026*, doi: [10.63317/2n39qzvk2eqe](https://doi.org/10.63317/2n39qzvk2eqe).
* Dataset: <https://huggingface.co/datasets/nurlanm/UniSkill_dataset> (Apache-2.0 license).

`source_manifest.json` lists the source files, the retrieval date and SHA-256 checksums. Every notebook verifies the checksum before running.

## Notebooks

| Notebook | Content |
|---|---|
| `LogisticRegression.ipynb` | Data cleaning, leakage-free grouped split, five lexical features, logistic regression |
| `02_RandomForest.ipynb` | Random forest on the same split and features |
| `03_GradientBoosting.ipynb` | Gradient boosting on the same split and features |
| `04_Model_Comparison_and_Hypothesis_Testing.ipynb` | All models and two baselines, McNemar tests, 30 repeated grouped splits, Friedman / Wilcoxon–Holm tests (H0/H1) |
| `05_Qualitative_Error_Analysis.ipynb` | Content analysis of misclassified pairs, codebook, tests of H2/H3, paper figures 1–2 |
| `06_Mentor_Mentee_Matching_Demo.ipynb` | Ranking of illustrative mentor profiles for mentee needs (paper figure 3) |

Run `04` before `05`, because `05` reads `results/primary_test_predictions.csv`. The other notebooks are independent.

## Main results

| Method | Mean macro-F1 over 30 grouped splits |
|---|---|
| Logistic regression | 0.678 ± 0.042 |
| Random forest | 0.655 ± 0.035 |
| Gradient boosting | 0.637 ± 0.045 |
| Cosine baseline | 0.628 |
| Majority baseline | 0.393 |

Friedman χ²(2) = 25.40, p < 0.001, Kendall’s W = 0.42. All pairwise differences remain significant after Holm correction.

## How to run

**Google Colab:** put the notebooks, `manual_records.csv` and `source_manifest.json` into one Google Drive folder and set `BASE` in the first code cell to that folder (default: `MyDrive/Assignment3_TulegenovaK`).

**Locally:** Python 3.9+ with `pandas`, `numpy`, `scipy`, `scikit-learn` (1.6.1 was used) and `matplotlib`. Start Jupyter in this folder.

```bash
pip install pandas numpy scipy scikit-learn==1.6.1 matplotlib jupyter
jupyter notebook
```

All outputs (metrics, predictions, error coding, statistics and figures) are saved to `results/`.

## Ethics

Only public data were used. No personal data of faculty members were processed. The mentor profiles in notebook 06 are illustrative and built from course texts.

## License

The code is released under the MIT License, see the LICENSE file. The data in manual_records.csv are derived from the UniSkill dataset by N. Musazade, J. Mezei and M. Zhang and remain under the Apache License 2.0.
