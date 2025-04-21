#!/usr/bin/env python
# coding: utf-8

# RUN WITH:
# caffeinate python rf_ea_tuning_simple_train_test.py | tee ea_tuned.log

# This script uses a genetic algorithm (GA) to find the best hyperparameters for a Random Forest regression model.
# It trains the model on podcast listening data and saves the final tuned model.

import datetime
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn_genetic import GASearchCV, ExponentialAdapter
from sklearn_genetic.space import Integer, Categorical

# Ignore some specific warnings
warnings.filterwarnings("ignore", message=".*ChildProcessError.*")

# Define how much of the data will be used as the test set
test_split_size = 0.10


def load_data(filename: str) -> pd.DataFrame:
    # Load the data and convert text categories into numbers
    p = Path(f"./data/{filename}")
    assert p.exists()

    train_df = pd.read_csv(filepath_or_buffer=p)

    # Convert categorical columns to numeric codes
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


# Load training data
train_df = load_data("train.csv")

# Separate features and target
X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_size, random_state=42
)

# These adapters adjust mutation/crossover rates during evolution
mutation_adapter = ExponentialAdapter(
    initial_value=0.8, end_value=0.2, adaptive_rate=0.1
)
crossover_adapter = ExponentialAdapter(
    initial_value=0.2, end_value=0.8, adaptive_rate=0.1
)

# Define the hyperparameter search space for the Random Forest
param_grid = {
    "n_estimators": Integer(160, 280),  # number of trees
    "max_depth": Integer(26, 34),  # how deep trees can grow
    "min_samples_split": Integer(2, 10),  # minimum samples to split a node
    "min_samples_leaf": Integer(1, 5),  # minimum samples at a leaf node
    "max_features": Categorical(
        ["sqrt", "log2", None]
    ),  # features to consider at each split
    "bootstrap": Categorical([True, False]),  # whether to bootstrap samples
}

# Define the model and GA search configuration
rf = RandomForestRegressor(random_state=42)
cv = 3  # number of folds for cross-validation

# Set up the genetic algorithm search
evolved_estimator = GASearchCV(
    estimator=rf,
    cv=cv,
    scoring="neg_root_mean_squared_error",  # we want to minimize RMSE
    param_grid=param_grid,
    n_jobs=-1,
    verbose=True,
    population_size=10,  # number of models per generation
    generations=6,  # number of generations to evolve
    mutation_probability=mutation_adapter,
    crossover_probability=crossover_adapter,
)

# Start the hyperparameter search
evolved_estimator.fit(X_train, y_train)

# Use the best model found during the search
model = evolved_estimator.best_estimator_

# Predict on test data and evaluate with RMSE
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# Retrain best model on all data to prepare for deployment
final_model = RandomForestRegressor(**evolved_estimator.best_params_, random_state=42)
final_model.fit(X, y)
model = final_model

# Show which features were most important in predictions
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))

# Define where to save the model bundle
model_path = Path("./models/rf_ga_tuned_fst_model_simple_train_test_bundle.pkl")

# Save model and all important info for later reuse
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
        "model_type": "RandomForestRegressor",
        "features": list(X.columns),
        "target": "Listening_Time_minutes",
    },
}

joblib.dump(model_bundle, model_path)
