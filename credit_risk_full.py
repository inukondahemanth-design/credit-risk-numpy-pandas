"""
Credit Risk Analysis with NumPy & pandas: the complete code in one file.
Run:  python3 credit_risk_full.py
(The notebooks 01-04 contain the same code, with explanations and outputs.)
"""
import numpy as np
import pandas as pd

# ======================================================================
# PART 1: EXPLORE THE DATA
# ======================================================================

# Load the data into a table (DataFrame) called df
df = pd.read_csv("data/german_credit.csv")

# Show the first 5 rows (5 is the default)
print(df.head())

# shape is an attribute (a stored fact), so no brackets
print(df.shape)

# The type of values in every column
print(df.dtypes)

# How many columns of each type
print(df.dtypes.value_counts())

# isnull() marks every empty cell as True; sum() counts the Trues per column
print(df.isnull().sum())

# Count how often each value appears in the class column
print(df["class"].value_counts())

# The same as percentages
print(df["class"].value_counts(normalize=True))

# count, mean, std, min, quartiles (50% = median) and max of every numeric column
print(df.describe())

# Take the age column out of pandas as a plain NumPy array
ages = df["age"].to_numpy()

print("Average:", ages.mean())
print("Median: ", np.median(ages))
print("Std:    ", ages.std())
print("Youngest:", ages.min(), "| Oldest:", ages.max())

# ages < 25 gives True/False per customer; the mean of True/False = share of True
print("Share under 25:", (ages < 25).mean())


# ======================================================================
# PART 2: CLEAN THE DATA + NEW FEATURES
# ======================================================================

df = pd.read_csv("data/german_credit.csv")
print(df.head(3))

# For every coded column: a dictionary {code: meaning}, taken from DATA_DICTIONARY.md
code_meanings = {
 "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no account"},
 "credit_history": {"A30": "no credits / all paid", "A31": "all paid at this bank", "A32": "paid on time so far", "A33": "delays in the past", "A34": "critical account"},
 "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/TV", "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "other"},
 "savings": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "unknown / none"},
 "employment": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"},
 "personal_status": {"A91": "male, divorced/separated", "A92": "female, divorced/separated/married", "A93": "male, single", "A94": "male, married/widowed", "A95": "female, single"},
 "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
 "property": {"A121": "real estate", "A122": "savings agreement / life insurance", "A123": "car or other", "A124": "unknown / none"},
 "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
 "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
 "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident", "A173": "skilled employee", "A174": "management / highly qualified"},
 "telephone": {"A191": "none", "A192": "yes"},
 "foreign_worker": {"A201": "yes", "A202": "no"},
}

# Go through every coded column and replace the codes with their meaning
for column, meanings in code_meanings.items():
    df[column] = df[column].map(meanings)

print(df.head(3))

# Total number of empty cells in the whole table (sum per column, then sum of those)
print(df.isnull().sum().sum())

# class == 2 gives True/False; .astype(int) turns True -> 1, False -> 0
df["is_bad"] = (df["class"] == 2).astype(int)

# Vectorized division: one line for all 1,000 customers, no loop
df["monthly_payment"] = (df["credit_amount"] / df["duration"]).round(1)

# pd.cut puts each age into a bin: (18, 25], (25, 35], (35, 50], (50, 100]
df["age_group"] = pd.cut(df["age"], bins=[18, 25, 35, 50, 100], labels=["19-25", "26-35", "36-50", "51+"])

# A comparison gives True/False for every customer
df["long_loan"] = df["duration"] > 24

print(df[["credit_amount", "duration", "monthly_payment", "age", "age_group", "long_loan", "class", "is_bad"]].head())

df.to_csv("data/german_credit_clean.csv", index=False)
print("Saved:", df.shape)


# ======================================================================
# PART 3: WHICH CUSTOMERS ARE RISKY?
# ======================================================================

df = pd.read_csv("data/german_credit_clean.csv")

# The overall bad rate, to compare every group against
overall = df["is_bad"].mean()
print(f"Overall bad rate: {overall:.0%}")

def bad_rate(column):
    """Bad rate and number of customers for every group in a column, riskiest first."""
    result = df.groupby(column, observed=True)["is_bad"].agg(["mean", "count"])
    result = result.rename(columns={"mean": "bad_rate", "count": "customers"})
    return result.sort_values("bad_rate", ascending=False).round(2)

print(bad_rate("checking_status"), "\n")

print(bad_rate("long_loan"), "\n")

print(bad_rate("age_group"), "\n")

print(bad_rate("savings"), "\n")

print(bad_rate("purpose"), "\n")

print(bad_rate("employment"), "\n")

print(bad_rate("housing"), "\n")

print(bad_rate("credit_history"), "\n")

print(df.groupby("is_bad")[["credit_amount", "duration", "monthly_payment", "age"]].mean().round(1))


# ======================================================================
# PART 4: LOGISTIC REGRESSION FROM SCRATCH (NumPy)
# ======================================================================

df = pd.read_csv("data/german_credit_clean.csv")
print(df.shape)

numeric_features = ["duration", "credit_amount", "installment_rate", "residence_since",
                    "age", "existing_credits", "num_dependents"]
categorical_features = ["checking_status", "credit_history", "purpose", "savings", "employment",
                        "other_debtors", "property", "other_installment_plans", "housing", "job"]

# One-hot encode the text columns; dtype=float gives 0.0/1.0 instead of True/False
X_table = pd.get_dummies(df[numeric_features + categorical_features],
                         columns=categorical_features, dtype=float)

# From here on we only use NumPy arrays
X = X_table.to_numpy()                 # features: one row per customer
y = df["is_bad"].to_numpy(dtype=float) # target: 1 = bad, 0 = good
feature_names = X_table.columns

print("X shape:", X.shape)   # (customers, features)
print("y shape:", y.shape)

rng = np.random.default_rng(seed=42)
shuffled = rng.permutation(len(X))      # shuffled row numbers
cut = int(0.8 * len(X))                 # 800

train_rows, test_rows = shuffled[:cut], shuffled[cut:]
X_train, X_test = X[train_rows], X[test_rows]   # fancy indexing: pick rows by a list of positions
y_train, y_test = y[train_rows], y[test_rows]

print("Train:", X_train.shape, "| Test:", X_test.shape)
print(f"Bad share in test data: {y_test.mean():.1%}")

mean = X_train.mean(axis=0)   # axis=0: one mean per column (feature)
std = X_train.std(axis=0)
std[std == 0] = 1             # avoid dividing by zero for a column that is constant in training

# Broadcasting: a (800, 53) array minus a (53,) array -> each column gets its own mean subtracted
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("Means after scaling (should be ~0):", X_train.mean(axis=0)[:3].round(3))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y_true, p):
    eps = 1e-12   # tiny number so we never take log(0)
    return -np.mean(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps))

# Quick check: sigmoid turns any number into a probability
print(sigmoid(np.array([-5, 0, 5])).round(3))

n_features = X_train.shape[1]
w = np.zeros(n_features)   # start with all weights at 0
b = 0.0                    # bias (the starting score)
learning_rate = 0.1

for step in range(2001):
    p = sigmoid(X_train @ w + b)        # 1. predicted probability for all 800 customers at once
    error = p - y_train                 # 2. how wrong each prediction is

    # 3. gradients: how much each weight contributed to the error
    grad_w = X_train.T @ error / len(y_train)
    grad_b = error.mean()

    # 4. take a small step downhill
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    if step % 500 == 0:
        print(f"step {step:4d} | loss {loss(y_train, p):.4f}")

def evaluate(threshold):
    p_test = sigmoid(X_test @ w + b)
    pred = (p_test >= threshold).astype(float)

    tp = np.sum((pred == 1) & (y_test == 1))
    fp = np.sum((pred == 1) & (y_test == 0))
    fn = np.sum((pred == 0) & (y_test == 1))
    tn = np.sum((pred == 0) & (y_test == 0))

    print(f"Threshold {threshold}")
    print(f"  Accuracy : {(tp + tn) / len(y_test):.1%}")
    print(f"  Precision: {tp / (tp + fp):.1%}")
    print(f"  Recall   : {tp / (tp + fn):.1%}   (caught {tp} of {tp + fn} bad customers)")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

baseline = 1 - y_test.mean()   # accuracy of always guessing "good"
print(f"Baseline (always 'good'): {baseline:.1%}\n")
evaluate(0.5)

train_acc = ((sigmoid(X_train @ w + b) >= 0.5) == y_train).mean()
print(f"\nTraining accuracy: {train_acc:.1%}")

evaluate(0.3)

weights = pd.Series(w, index=feature_names).sort_values()

print("Strongest signals of LOW risk:")
print(weights.head(5).round(2))
print("\nStrongest signals of HIGH risk:")
print(weights.tail(5).round(2))
