#!/usr/bin/env python
# coding: utf-8

# RUN WITH:
# caffeinate python rf_ea_tuning_simple_train_test.py | tee ea_tuned.log

# In[2]:


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

warnings.filterwarnings("ignore", message=".*ChildProcessError.*")


test_split_size = 0.25
# In[4]:


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


train_df = load_data("train.csv")#.sample(n=1000)


# In[ ]:


X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_size, random_state=42
)


# In[2]:


mutation_adapter = ExponentialAdapter(
    initial_value=0.1, end_value=0.05, adaptive_rate=0.2
)
crossover_adapter = ExponentialAdapter(
    initial_value=0.5, end_value=0.8, adaptive_rate=0.2
)

param_grid = {
    "n_estimators": Integer(160, 280),
    "max_depth": Integer(26, 34),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 5),
    "max_features": Categorical(["sqrt", "log2", None]),
    "bootstrap": Categorical([True, False]),
}

rf = RandomForestRegressor(random_state=42)
cv = 2

evolved_estimator = GASearchCV(
    estimator=rf,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    param_grid=param_grid,
    n_jobs=-1,
    verbose=True,
    population_size=4,
    generations=4,
    mutation_probability=mutation_adapter,
    crossover_probability=crossover_adapter,
)

evolved_estimator.fit(X_train, y_train)
model = evolved_estimator.best_estimator_


# In[ ]:


y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# Retrain on full dataset using the best found parameters
final_model = RandomForestRegressor(**evolved_estimator.best_params_, random_state=42)
final_model.fit(X, y)
model = final_model


# In[ ]:


importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))


# In[ ]:


model_path = Path("./models/rf_ga_tuned_fst_model_simple_train_test_bundle.pkl")


# In[ ]:


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
