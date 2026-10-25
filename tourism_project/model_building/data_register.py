"""
data_register.py
================
Step 1 of the MLOps pipeline - DATA REGISTRATION.

What this script does
---------------------
1. Locates the raw dataset that lives inside the GitHub repository
   (tourism_project/data/tourism.csv). The repo itself is our "data registry",
   so no external dataset store is needed.
2. Validates the dataset against the expected schema (the data dictionary):
     - every expected column must be present,
     - the target column must be binary (0/1),
     - the file must not be empty.
3. Prints a readable summary (shape, dtypes, missing values, duplicates,
   target distribution, categorical levels, numeric statistics).
4. Exits with a NON-ZERO status code if validation fails, which makes the
   GitHub Actions job fail and stops all downstream jobs (data-prep, training).
   This is a simple but effective "data quality gate".

Run from the repository root:
    python tourism_project/model_building/data_register.py
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path of the dataset relative to the repository root (the workflow runs from
# the repo root, and so does the Colab notebook).
DATA_PATH = Path("tourism_project/data/tourism.csv")

# Target variable as defined in the data dictionary
TARGET = "ProdTaken"

# Expected columns taken directly from the project's data dictionary.
EXPECTED_COLUMNS = [
    # Customer details
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    # Customer interaction data
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch",
]


def print_header(title: str) -> None:
    """Print a visually separated section header for readable CI logs."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate(df: pd.DataFrame) -> list:
    """
    Run schema / quality checks on the dataframe.

    Returns a list of error messages. An empty list means the dataset passed.
    """
    errors = []

    # Check 1: the file must contain rows
    if df.empty:
        errors.append("Dataset is empty.")

    # Check 2: all expected columns must be present
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    # Check 3: the target must exist and be binary (0/1)
    if TARGET in df.columns:
        target_values = set(df[TARGET].dropna().unique())
        if not target_values.issubset({0, 1}):
            errors.append(f"Target '{TARGET}' is not binary: {target_values}")

    return errors


def main() -> None:
    print_header("STEP 1: DATA REGISTRATION & VALIDATION")

    # ------------------------------------------------------------------
    # Load the dataset from the repository data folder
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at {DATA_PATH.resolve()}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset registered from repository path : {DATA_PATH}")
    print(f"File size                               : "
          f"{DATA_PATH.stat().st_size / 1024:.1f} KB")

    # ------------------------------------------------------------------
    # Validate schema
    # ------------------------------------------------------------------
    print_header("Schema validation")
    errors = validate(df)
    for col in EXPECTED_COLUMNS:
        status = "OK " if col in df.columns else "MISSING"
        print(f"  [{status}] {col}")

    # Columns that are in the file but not in the data dictionary
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_cols:
        print(f"\n  Extra (non-dictionary) columns found: {extra_cols}")
        print("  -> These will be dropped during data preparation.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("Dataset summary")
    print(f"Rows               : {df.shape[0]}")
    print(f"Columns            : {df.shape[1]}")
    print(f"Duplicate rows     : {df.duplicated().sum()}")
    print(f"Total missing cells: {int(df.isna().sum().sum())}")

    print("\nColumn data types and missing values:")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique_values": df.nunique(),
    })
    print(info.to_string())

    if TARGET in df.columns:
        print(f"\nTarget distribution ({TARGET}):")
        counts = df[TARGET].value_counts().sort_index()
        pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
        for label in counts.index:
            print(f"  {label}: {counts[label]:>5} rows ({pct[label]:.1f}%)")

    print("\nCategorical columns and their levels:")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        print(f"  {col:<15}: {sorted(df[col].dropna().unique().tolist())}")

    print("\nNumeric summary statistics:")
    print(df.describe().T.round(2).to_string())

    # ------------------------------------------------------------------
    # Quality gate: fail the job if validation failed
    # ------------------------------------------------------------------
    print_header("Validation result")
    if errors:
        for err in errors:
            print(f"  FAILED: {err}")
        sys.exit(1)

    print("  PASSED - all expected columns are present and the target is binary.")
    print("  Dataset successfully registered for the pipeline.")


if __name__ == "__main__":
    main()
