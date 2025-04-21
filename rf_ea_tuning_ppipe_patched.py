#!/usr/bin/env python
# coding: utf-8

# RUN WITH:
# caffeinate python rf_ea_tuning_ppipe_patched.py | tee logs/ea_pp_tuned.log

import datetime
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn_genetic import GASearchCV, ExponentialAdapter
from sklearn_genetic.space import Integer, Categorical

from optional_transformer import OptionalTransformer

warnings.filterwarnings("ignore", message=".*ChildProcessError.*")


test_split_size = 0.05


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


train_df = load_data("train.csv")#.sample(n=100)

X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_size, random_state=42
)


pipeline = Pipeline(
    [
        ("imputer", OptionalTransformer()),
        ("scaler", OptionalTransformer()),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

mutation_adapter = ExponentialAdapter(
    initial_value=0.8, end_value=0.2, adaptive_rate=0.1
)
crossover_adapter = ExponentialAdapter(
    initial_value=0.2, end_value=0.8, adaptive_rate=0.1
)

param_grid = {
    "imputer__transformer": Categorical(
        [None, SimpleImputer(strategy="mean"), SimpleImputer(strategy="median")]
    ),
    "scaler__transformer": Categorical([None, MinMaxScaler()]),
    "regressor__n_estimators": Integer(160, 280),
    "regressor__max_depth": Integer(26, 34),
    "regressor__min_samples_split": Integer(2, 10),
    "regressor__min_samples_leaf": Integer(1, 5),
    "regressor__max_features": Categorical(["sqrt", "log2", None]),
    "regressor__bootstrap": Categorical([True, False]),
}

cv = 2
evolved_estimator = GASearchCV(
    estimator=pipeline,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    param_grid=param_grid,
    n_jobs=-1,
    verbose=True,
    population_size=10,
    generations=6,
    mutation_probability=mutation_adapter,
    crossover_probability=crossover_adapter,
)

evolved_estimator.fit(X_train, y_train)
model = evolved_estimator.best_estimator_

y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# Retrain on full dataset using the best found parameters
final_model = Pipeline(
    [
        (
            "imputer",
            OptionalTransformer(evolved_estimator.best_params_["imputer__transformer"]),
        ),
        (
            "scaler",
            OptionalTransformer(evolved_estimator.best_params_["scaler__transformer"]),
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=evolved_estimator.best_params_["regressor__n_estimators"],
                max_depth=evolved_estimator.best_params_["regressor__max_depth"],
                min_samples_split=evolved_estimator.best_params_[
                    "regressor__min_samples_split"
                ],
                min_samples_leaf=evolved_estimator.best_params_[
                    "regressor__min_samples_leaf"
                ],
                max_features=evolved_estimator.best_params_["regressor__max_features"],
                bootstrap=evolved_estimator.best_params_["regressor__bootstrap"],
                random_state=42,
            ),
        ),
    ]
)
final_model.fit(X, y)
model = final_model

model_path = Path("./models/rf_ga_tuned_fst_model_with_pipeline.pkl")

model_bundle = {
    "model": model,
    "metrics": {
        "rmse": rmse,
    },
    "test_size": test_split_size,
    "best_params": evolved_estimator.best_params_,
    "param_grid": param_grid,
    "metadata": {
        "trained_on": str(datetime.datetime.now()),
        "model_type": "Pipeline(RandomForestRegressor)",
        "features": list(X.columns),
        "target": "Listening_Time_minutes",
    },
}

joblib.dump(model_bundle, model_path)
