import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


# Part 1. Initialize data and train test split

data = pd.read_csv("../feature_processing/all_data.csv")
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

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size= 0.15, random_state = 44)

# Use the MinMaxScaler on curvature and rpm and the encoder on strings
cat_features = X_train.select_dtypes(include="object").columns.tolist()
num_features = X_train.select_dtypes(include="float").columns.tolist()

diff_geo_cols = ["speed_curvature_corr","mean_curvature","max_curvature"]
other = [c for c in num_features if c not in diff_geo_cols]

t_weights = {"diff_geo_cols" :3.0 , "other_num" : 1.0, "cat" : 0.5}

preprocessor = ColumnTransformer(
    transformers = [
        ("diff_geo_cols", MinMaxScaler(), diff_geo_cols),
        ("other_num", MinMaxScaler(), other),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output = False), cat_features)
    ],
    transformer_weights = t_weights,
).set_output(transform="pandas")


model_pipeline = Pipeline(
    steps= [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ]
)

print(y_test.value_counts(normalize=True))


with mlflow.start_run(run_name="logreg_baseline"):
    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    probabilities = model_pipeline.predict_proba(X_test)
    score = model_pipeline.score(X_test, y_test)
    print(f"Model Score: {score}")

    mlflow.log_metric("accuracy", score)
    
    preprocessor = model_pipeline.named_steps["preprocessor"]
    clf = model_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "group": [n.split("__", 1)[0] for n in feature_names],
        "raw_coef": clf.coef_[0],
    })

    # Undo the transformer weight so coefficients are comparable across groups
    feature_importance["weight"] = (
        feature_importance["group"].map(preprocessor.transformer_weights).fillna(1.0)
    )
    feature_importance["Coefficient"] = feature_importance["raw_coef"] / feature_importance["weight"]

    feature_importance["abs_impact"] = feature_importance["Coefficient"].abs()
    feature_importance = (
        feature_importance
        .sort_values("abs_impact", ascending=False)
        .drop(columns="abs_impact")
    )

    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    mlflow.log_params({"transform_weights" : t_weights})
    mlflow.log_params({"test_size": 0.15, "classifier": "LogisticRegression"})
    

    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        name=f"f1_top10_logreg",             
        input_example=X_train.head(),
        registered_model_name=f"f1-top10-predictor",  # optional
    )