# Credit Risk Analysis with NumPy & pandas

**Which bank customers fail to pay back their loans?**
Analysing 1,000 real loan applicants from a German bank and predicting **good** vs **bad** credit risk,
using **NumPy and pandas only**. The prediction model is built **from scratch in NumPy**.

**Dataset:** [Statlog German Credit Data (UCI)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (CC BY 4.0): 1,000 applicants, 20 features, target `class` (1 = good, 2 = bad). Code meanings: [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Project structure
| Notebook | What it does |
|---|---|
| `01_explore.ipynb` | First look: size, columns, types, missing values, target |
| `02_clean.ipynb` | Decode `A11`-style codes into readable categories, create new features |
| `03_analysis.ipynb` | Which groups are riskiest? (groupby, bad rates) |
| `04_model_from_scratch.ipynb` | Logistic regression in pure NumPy: scaling, one-hot encoding, train/test split, gradient descent, accuracy |

## Key findings
- **30% of customers are bad**, so any model must beat a 70% "always good" baseline.
- **Negative bank account** is the strongest warning sign: **49% bad** (vs 12% for customers without a checking account).
- **Long loans (> 24 months): 44% bad**, compared with 26% for shorter loans.
- **Young customers (19–25): 42% bad**, the highest of all age groups.
- **Low savings (< 100 DM): 36% bad**, vs 13% with ≥ 1,000 DM.
- Surprise: customers with a "critical" credit history have the **lowest** bad rate (17%), likely because the bank was already stricter with them.
- Sex and nationality are in the data but are **not used** as decision factors (fairness / EU law).

## Model results (logistic regression from scratch in NumPy)
| | Accuracy | Precision (bad) | Recall (bad) |
|---|---|---|---|
| Baseline: always "good" | 65.5% | – | 0% |
| Model, threshold 0.5 | **71.0%** | 61.2% | 43.5% |
| Model, threshold 0.3 | 66.5% | 51.1% | **65.2%** |

Missing a bad customer is far more costly for a bank than rejecting a good one, so the lower threshold catches more bad loans at a small cost in accuracy.
The model's strongest risk signals (loan duration, a negative bank account) match the manual analysis.

## How to run
```bash
pip install numpy pandas jupyter
jupyter notebook 01_explore.ipynb
```
