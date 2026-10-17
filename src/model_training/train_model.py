import datetime as dt
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# -------------------------------------------------------------------------
# Run mode: MLflow locally, plain files on CI
# ---------------------------------------------------------------------------
# GitHub Actions sets GITHUB_ACTIONS=true automatically. To test the remote
# path on your machine, run:  TRAIN_MODE=remote make train
# (TRAIN_MODE=local forces MLflow even on CI.)


def is_remote_run() -> bool:
    mode = os.getenv("TRAIN_MODE", "").lower()
    if mode in {"remote", "local"}:
        return mode == "remote"
    return os.getenv("GITHUB_ACTIONS") == "true"


# Created next to this script, e.g. src/model_training/artifacts/remote_workflows
REMOTE_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "remote_workflows"

TEST_SIZE = 0.15


# ---------------------------------------------------------------------------
# Part 1. Load data, clean it and split
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parents[1] / "feature_processing" / "all_data.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Run `make create-dataset` first."
    )

data = pd.read_csv(DATA_PATH)
y_data = data["top10"]
x_data = data.drop(columns=["top10"])
num_cols = x_data.select_dtypes(include="number").columns
inf_mask = np.isinf(x_data[num_cols]).any(axis=1)
print(inf_mask.sum(), "rows with infinities")
nan_mask = x_data.isna().any(axis=1)
print(nan_mask.sum(), "rows with NaN")
keep = ~(inf_mask | nan_mask)
x_data = x_data[keep]
y_data = y_data[keep]
y_data = y_data.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=TEST_SIZE, random_state=44
)

# ---------------------------------------------------------------------------
# Part 2. Preprocessing and model
# ---------------------------------------------------------------------------

# Use the MinMaxScaler on numeric columns and the encoder on strings
cat_features = X_train.select_dtypes(include="object").columns.tolist()
num_features = X_train.select_dtypes(include="float").columns.tolist()

diff_geo_cols = ["speed_curvature_corr", "mean_curvature", "max_curvature"]
other = [c for c in num_features if c not in diff_geo_cols]

t_weights = {"diff_geo_cols": 3.0, "other_num": 1.0, "cat": 0.5}

preprocessor = ColumnTransformer(
    transformers=[
        ("diff_geo_cols", MinMaxScaler(), diff_geo_cols),
        ("other_num", MinMaxScaler(), other),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ],
    transformer_weights=t_weights,
).set_output(transform="pandas")

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)

print(y_test.value_counts(normalize=True))

# ---------------------------------------------------------------------------
# Part 3. Train, evaluate and compute feature importance
# ---------------------------------------------------------------------------

model_pipeline.fit(X_train, y_train)

predictions = model_pipeline.predict(X_test)
probabilities = model_pipeline.predict_proba(X_test)
score = model_pipeline.score(X_test, y_test)
print(f"Model Score: {score}")

fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
clf = model_pipeline.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "group": [n.split("__", 1)[0] for n in feature_names],
        "raw_coef": clf.coef_[0],
    }
)

# Undo the transformer weight so coefficients are comparable across groups
feature_importance["weight"] = (
    feature_importance["group"].map(fitted_preprocessor.transformer_weights).fillna(1.0)
)
feature_importance["Coefficient"] = feature_importance["raw_coef"] / feature_importance["weight"]

feature_importance["abs_impact"] = feature_importance["Coefficient"].abs()
feature_importance = (
    feature_importance.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")
)

metrics = {
    "accuracy": float(score),
    "test_size": TEST_SIZE,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "classifier": "LogisticRegression",
    "transformer_weights": t_weights,
    "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "sklearn_version": sklearn.__version__,
    "git_sha": os.getenv("GITHUB_SHA"),  # None when run locally
}

# ---------------------------------------------------------------------------
# Part 4. Save results
# ---------------------------------------------------------------------------


def log_to_mlflow() -> None:
    """Local run: track everything in MLflow and register the model."""
    # Use a dedicated active experiment instead of relying on MLflow's deleted
    # default experiment in an existing tracking database.
    mlflow.set_experiment("f1-top10-training")

    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.log_metric("accuracy", metrics["accuracy"])

        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        mlflow.log_params({"transform_weights": metrics["transformer_weights"]})
        mlflow.log_params(
            {"test_size": metrics["test_size"], "classifier": metrics["classifier"]}
        )

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="f1_top10_logreg",
            input_example=X_train.head(),
            registered_model_name="f1-top10-predictor",
        )


def dump_to_disk() -> None:
    """Remote run (CI): no MLflow server, so write plain files."""
    REMOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = REMOTE_OUTPUT_DIR / "model_pipeline.joblib"
    joblib.dump(model_pipeline, model_path)

    metrics_path = REMOTE_OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    feature_importance.to_csv(REMOTE_OUTPUT_DIR / "feature_importance.csv", index=False)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if is_remote_run():
    print("Remote run detected: skipping MLflow and writing files instead.")
    dump_to_disk()
else:
    log_to_mlflow()