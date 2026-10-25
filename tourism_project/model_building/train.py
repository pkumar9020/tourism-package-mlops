"""
train.py
========
Step 3 of the MLOps pipeline - MODEL BUILDING WITH EXPERIMENT TRACKING.

What this script does
---------------------
1. Loads the train / test splits produced by prep.py. In GitHub Actions these
   files are DOWNLOADED FROM THE WORKFLOW ARTIFACT created by the data-prep job.
2. Builds a single scikit-learn Pipeline:
       preprocessing (impute + scale numeric, impute + one-hot encode text)
       -> XGBoost classifier
   Keeping preprocessing inside the pipeline means the saved model file
   contains everything needed to score raw customer data in the Streamlit app.
3. Defines a hyperparameter grid and tunes the model with GridSearchCV
   (5-fold stratified cross-validation, optimising F1 score).
4. Logs to MLflow:
     - one NESTED run per hyperparameter combination (all tuned parameters
       plus the cross-validated scores),
     - the best parameters, train/test metrics, confusion matrix,
       feature importances and the model itself in the parent run.
5. Evaluates the best model on the held-out test set.
6. Saves the best model to tourism_project/deployment/ so the workflow can
   COMMIT it to the repository, where Streamlit Community Cloud picks it up.

Why XGBoost + F1?
-----------------
Only ~19% of customers buy the package (class imbalance). Accuracy would look
good even for a model that predicts "No" for everyone, so we optimise F1
(balance of precision and recall) and use scale_pos_weight to give buyers
more weight during training.

Run from the repository root:
    python tourism_project/model_building/train.py
"""

import json
import os
import warnings

# MLflow warns about integer columns in the input example; it is informational
# only (our features never arrive with missing values at prediction time).
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Keep CI / notebook logs readable: stop MLflow printing a "View run" URL for
# every one of the 48 nested runs (must be set before mlflow is imported).
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
import socket
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "."))                 # where the splits are
MODEL_DIR = Path("tourism_project/deployment")              # committed by CI
MODEL_PATH = MODEL_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
EXPERIMENT_NAME = "tourism-wellness-package-prediction"
RANDOM_STATE = 42
MIN_TEST_F1 = 0.40   # quality gate: below this the pipeline fails (no deploy)

# Feature groups (must match the columns produced by prep.py)
NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Hyperparameter grid for XGBoost (2 x 3 x 2 x 2 x 2 = 48 combinations,
# x 5 folds = 240 fits; small dataset so this runs in a couple of minutes).
PARAM_GRID = {
    "model__n_estimators": [100, 200],      # number of boosted trees
    "model__max_depth": [3, 5, 7],          # tree depth (model complexity)
    "model__learning_rate": [0.05, 0.1],    # shrinkage per tree
    "model__colsample_bytree": [0.6, 0.8],  # share of features per tree
    "model__subsample": [0.8, 1.0],         # share of rows per tree
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def configure_mlflow() -> str:
    """
    Point MLflow at a tracking server.

    - In GitHub Actions, the workflow starts an MLflow server on
      http://localhost:5000 and passes it via MLFLOW_TRACKING_URI.
    - In Colab, the notebook starts the same server (optionally exposed via
      ngrok so you can browse the UI).
    - If no server is reachable we fall back to a local ./mlruns_local folder, so
      training NEVER fails just because the tracking UI is down.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            socket.create_connection((parsed.hostname, port), timeout=3).close()
        except OSError:
            print(f"MLflow server not reachable at {uri}; using local ./mlruns_local")
            uri = Path("mlruns_local").resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI : {uri}")
    print(f"MLflow experiment   : {EXPERIMENT_NAME}")
    return uri


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Preprocessing + XGBoost wrapped in one Pipeline object."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # robust to outliers
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": an unseen category in production will not
        # crash the app; it is simply encoded as all zeros.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,  # counter the class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Standard binary-classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("STEP 3: MODEL BUILDING WITH EXPERIMENT TRACKING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load the splits (downloaded from the workflow artifact in CI)
    # ------------------------------------------------------------------
    X_train = pd.read_csv(DATA_DIR / "Xtrain.csv")
    X_test = pd.read_csv(DATA_DIR / "Xtest.csv")
    y_train = pd.read_csv(DATA_DIR / "ytrain.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "ytest.csv").squeeze("columns")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Weight for the positive class = (#negatives / #positives)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Class imbalance -> scale_pos_weight = {scale_pos_weight:.2f}")

    # ------------------------------------------------------------------
    # 2. Define model + hyperparameter search
    # ------------------------------------------------------------------
    pipeline = build_pipeline(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring={"f1": "f1", "recall": "recall",
                 "precision": "precision", "roc_auc": "roc_auc"},
        refit="f1",            # pick the best model by cross-validated F1
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    configure_mlflow()

    with mlflow.start_run(run_name="xgboost_gridsearch") as parent_run:
        # Context parameters for traceability
        mlflow.log_params({
            "model_type": "XGBClassifier",
            "cv_folds": 5,
            "tuning_metric": "f1",
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
            "scale_pos_weight": round(scale_pos_weight, 4),
        })
        mlflow.set_tag("pipeline_step", "model_training")
        mlflow.set_tag("git_commit", os.getenv("GITHUB_SHA", "local"))

        # --------------------------------------------------------------
        # 3. Tune
        # --------------------------------------------------------------
        n_combos = int(pd.Series([len(v) for v in PARAM_GRID.values()]).prod())
        print(f"\nTuning {n_combos} hyperparameter combinations x 5 folds ...")
        grid_search.fit(X_train, y_train)

        # --------------------------------------------------------------
        # 4. Log EVERY tuned combination as a nested MLflow run
        # --------------------------------------------------------------
        results = pd.DataFrame(grid_search.cv_results_)
        for i in range(len(results)):
            params = results.loc[i, "params"]
            with mlflow.start_run(run_name=f"combo_{i:02d}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v
                                   for k, v in params.items()})
                mlflow.log_metrics({
                    "cv_mean_f1": results.loc[i, "mean_test_f1"],
                    "cv_std_f1": results.loc[i, "std_test_f1"],
                    "cv_mean_recall": results.loc[i, "mean_test_recall"],
                    "cv_mean_precision": results.loc[i, "mean_test_precision"],
                    "cv_mean_roc_auc": results.loc[i, "mean_test_roc_auc"],
                })
        print(f"Logged {len(results)} nested runs (one per combination).")

        # Show the top-5 combinations in the logs
        top5 = results.sort_values("rank_test_f1").head(5)
        cols = ["params", "mean_test_f1", "mean_test_recall", "mean_test_roc_auc"]
        print("\nTop 5 combinations by cross-validated F1:")
        print(top5[cols].to_string(index=False))

        # Best parameters -> parent run
        best_params = {k.replace("model__", ""): v
                       for k, v in grid_search.best_params_.items()}
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)
        print(f"\nBest parameters : {best_params}")
        print(f"Best CV F1      : {grid_search.best_score_:.4f}")

        # --------------------------------------------------------------
        # 5. Evaluate the best model on train and held-out test data
        # --------------------------------------------------------------
        best_model = grid_search.best_estimator_
        metrics = {}
        for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            proba = best_model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)
            split_metrics = compute_metrics(y, pred, proba)
            metrics[split] = split_metrics
            mlflow.log_metrics({f"{split}_{k}": v for k, v in split_metrics.items()})

        print("\nPerformance of the best model:")
        print(pd.DataFrame(metrics).T.round(4).to_string())

        y_test_pred = best_model.predict(X_test)
        print("\nClassification report (test set):")
        print(classification_report(y_test, y_test_pred,
                                    target_names=["Not purchased", "Purchased"]))
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        print(f"Confusion matrix (test): TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        mlflow.log_metrics({"test_tn": tn, "test_fp": fp,
                            "test_fn": fn, "test_tp": tp})

        # Feature importances (aggregated per original feature for readability)
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        importances = pd.Series(
            best_model.named_steps["model"].feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)
        importances.to_csv("feature_importances.csv", header=["importance"])
        mlflow.log_artifact("feature_importances.csv")
        print("\nTop 10 most important features:")
        print(importances.head(10).round(4).to_string())

        # --------------------------------------------------------------
        # 6. Quality gate, then save the model for deployment
        # --------------------------------------------------------------
        if metrics["test"]["f1"] < MIN_TEST_F1:
            raise SystemExit(
                f"Test F1 {metrics['test']['f1']:.3f} is below the quality gate "
                f"({MIN_TEST_F1}). Model will NOT be committed / deployed."
            )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump({
                "best_params": best_params,
                "best_cv_f1": round(grid_search.best_score_, 4),
                "metrics": {s: {k: round(v, 4) for k, v in m.items()}
                            for s, m in metrics.items()},
                "mlflow_run_id": parent_run.info.run_id,
                "git_commit": os.getenv("GITHUB_SHA", "local"),
            }, f, indent=2)
        print(f"\nSaved best model -> {MODEL_PATH}")
        print(f"Saved metrics    -> {METRICS_PATH}")

        # Also store the model and metrics inside MLflow (artifact store)
        mlflow.log_artifact(str(METRICS_PATH))
        try:
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                input_example=X_test.head(3),   # lets MLflow infer the signature
            )
        except Exception as exc:   # logging the model is a "nice to have"
            print(f"(MLflow model logging skipped: {exc})")

        print(f"\nMLflow parent run id: {parent_run.info.run_id}")
        print("Model training completed successfully.")


if __name__ == "__main__":
    main()
