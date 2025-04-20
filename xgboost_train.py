#!/usr/bin/env python
# coding: utf-8

# === Setup and Configuration ===
import datetime
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (KFold, ParameterSampler,
                                     RandomizedSearchCV, train_test_split)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", message=".*ChildProcessError.*")

test_split_size = 0.05

# === Data Loading and Preprocessing ===
def load_data(filename: str) -> pd.DataFrame:
    p = Path(f"./data/{filename}")
    assert p.exists()

    train_df = pd.read_csv(filepath_or_buffer=p)

    train_df["Genre"] = train_df.Genre.astype("category").cat.codes
    train_df["Episode_Sentiment"] = train_df.Episode_Sentiment.astype(
        "category"
    ).cat.codes
    train_df["Publication_Day"] = train_df.Publication_Day.astype("category").cat.codes
    train_df["Publication_Time"] = train_df.Publication_Time.astype(
        "category"
    ).cat.codes
    train_df["Episode_Title"] = train_df.Episode_Title.astype("category").cat.codes
    train_df["Podcast_Name"] = train_df.Podcast_Name.astype("category").cat.codes

    return train_df

train_df = load_data("train.csv")

# === Train/Test Split ===
X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_size, random_state=42
)

# === Hyperparameter Search Configuration ===
param_dist = {
    "n_estimators": [160, 180, 200, 220, 240, 260, 280, 300],
    "max_depth": [10, 15, 20, 25],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# === Manual Parameter Search with Early Stopping ===
param_sampler = list(ParameterSampler(param_dist, n_iter=42, random_state=42))

best_score = float("inf")
best_model = None
best_params = None

for i, params in enumerate(param_sampler):
    model = XGBRegressor(
        **params,
        random_state=i+1,
        verbosity=0,
        tree_method="hist",
        early_stopping_rounds=30,
        eval_metric="rmse"
    )

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.25, random_state=i+1
    )

    model.fit(
        X_train_split,
        y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False
    )

    y_val_pred = model.predict(X_val_split)
    score = root_mean_squared_error(y_val_split, y_val_pred)

    if score < best_score:
        best_score = score
        best_model = model
        best_params = params

model = best_model

# === Final Evaluation ===
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# === Retrain on Full Dataset with Best Params ===
X_train_full, X_val_final, y_train_full, y_val_final = train_test_split(X, y, test_size=0.05, random_state=42)
final_model = XGBRegressor(**best_params, random_state=42, verbosity=0, tree_method="hist", early_stopping_rounds=30, eval_metric="rmse")
final_model.fit(X_train_full, y_train_full, eval_set=[(X_val_final, y_val_final)], verbose=False)
model = final_model

# === Feature Importance ===
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))

# === Save Model Bundle ===
model_path = Path("./models/xgb_tuned_scnd_model_bundle.pkl")

model_bundle = {
    "model": model,
    "metrics": {
        "rmse": rmse,
    },
    "test_size": test_split_size,
    "best_params": best_params,
    "param_grid": param_dist,
    "metadata": {
        "trained_on": str(datetime.datetime.now()),
        "model_type": "XGBRegressor",
        "features": list(X.columns),
        "target": "Listening_Time_minutes",
    },
}

joblib.dump(model_bundle, model_path)
