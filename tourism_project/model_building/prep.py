"""
prep.py
=======
Step 2 of the MLOps pipeline - DATA PREPARATION.

What this script does
---------------------
1. Loads the registered dataset DIRECTLY from the repository data folder
   (tourism_project/data/tourism.csv).
2. Cleans the data:
     a. Drops columns that carry no predictive signal:
          - "Unnamed: 0" : a leftover pandas row index from an earlier export
          - "CustomerID" : a unique identifier (would only let the model memorise)
     b. Trims stray whitespace in all text columns.
     c. Fixes the data-entry typo "Fe Male" -> "Female" in Gender.
     d. Removes rows with a missing target (cannot be used for training).
     e. Removes exact duplicate rows (duplicates inflate some patterns and can
        leak between train and test).
   Missing-value imputation and encoding are deliberately NOT done here - they
   are learned inside the scikit-learn Pipeline in train.py, using training
   data only. This avoids data leakage and guarantees the deployed model
   applies the exact same preprocessing to new customers.
3. Splits the cleaned data into train (80%) and test (20%) sets using a
   STRATIFIED split, so both sets keep the same share of buyers (~19%).
4. Saves Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally. GitHub Actions
   uploads these files as a workflow artifact so the training job can
   download them.

Run from the repository root:
    python tourism_project/model_building/prep.py
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("tourism_project/data/tourism.csv")   # dataset inside the repo
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "."))         # where splits are saved
TARGET = "ProdTaken"
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID"]          # no predictive value
TEST_SIZE = 0.20                                        # 80/20 split
RANDOM_STATE = 42                                       # reproducibility


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules and print what changed at each step."""
    rows_before = len(df)

    # (a) Drop identifier / index columns (only the ones that actually exist)
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Dropped unnecessary columns          : {to_drop}")

    # (b) Trim whitespace in every text column
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # (c) Fix inconsistent category labels
    typo_count = int((df["Gender"] == "Fe Male").sum())
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    print(f"Fixed 'Fe Male' -> 'Female' in Gender : {typo_count} rows")

    # (d) Rows without a target cannot be used for supervised learning
    missing_target = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    print(f"Removed rows with missing target     : {missing_target}")

    # (e) Exact duplicate rows (checked AFTER dropping CustomerID, otherwise
    #     the unique ID would hide them)
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed exact duplicate rows         : {dup_count}")

    print(f"Rows: {rows_before} -> {len(df)} after cleaning")
    return df


def main() -> None:
    print("=" * 70)
    print("STEP 2: DATA PREPARATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load directly from the repository data folder
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH} -> shape {df.shape}\n")

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------
    df = clean_data(df)

    # Report remaining missing values (they will be imputed in train.py)
    remaining_missing = df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    print("\nRemaining missing values (imputed later inside the model pipeline):")
    print(remaining_missing.to_string() if not remaining_missing.empty else "  None")

    # Quick outlier report - kept intentionally: tree-based models are robust
    # to extreme values and these look like genuine (rare) customers.
    print("\nPotential outliers (values beyond 1.5 x IQR) - kept, trees are robust:")
    for col in ["DurationOfPitch", "MonthlyIncome", "NumberOfTrips"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        print(f"  {col:<16}: {int((df[col] > upper).sum())} rows above {upper:,.1f}")

    # ------------------------------------------------------------------
    # Split features / target, then stratified train-test split
    # ------------------------------------------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # keep the buyer ratio identical in both sets
    )

    print(f"\nTrain set: {X_train.shape[0]} rows | buyer rate {y_train.mean():.2%}")
    print(f"Test set : {X_test.shape[0]} rows | buyer rate {y_test.mean():.2%}")
    print(f"Features ({X.shape[1]}): {list(X.columns)}")

    # ------------------------------------------------------------------
    # Save splits locally -> uploaded as a workflow artifact by GitHub Actions
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "Xtrain.csv": X_train,
        "Xtest.csv": X_test,
        "ytrain.csv": y_train,
        "ytest.csv": y_test,
    }
    for name, data in outputs.items():
        path = OUTPUT_DIR / name
        data.to_csv(path, index=False)
        print(f"Saved {path}  ({len(data)} rows)")

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()
