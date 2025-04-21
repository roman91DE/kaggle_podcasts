#!/usr/bin/env python
# coding: utf-8

# RUN WITH:
# caffeinate python rf_tuning_imp.py | tee rf_pipe_tuned.log

# In[2]:


import datetime
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*ChildProcessError.*")


test_split_size = 0.1
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


train_df = load_data("train.csv")#.sample(n=100)


# In[ ]:


X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_size, random_state=42
)


# In[2]:


pipeline = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(random_state=42))
])

param_dist = {
    "imputer": ["passthrough", SimpleImputer(strategy="mean"), SimpleImputer(strategy="median")],
    "scaler": ["passthrough", StandardScaler()],
    "regressor__n_estimators": [160, 180, 200, 220, 240, 250],
    "regressor__max_depth": [None, 26, 28, 30, 32, 34],
    "regressor__min_samples_split": [2, 3],
    "regressor__min_samples_leaf": [1, 2],
    "regressor__max_features": ["sqrt"],
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=24,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)
model = random_search.best_estimator_


# In[ ]:


y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# Retrain on full dataset using the best found parameters
final_model = Pipeline([
    ("imputer", random_search.best_params_["imputer"]),
    ("scaler", random_search.best_params_["scaler"]),
    ("regressor", RandomForestRegressor(
        n_estimators=random_search.best_params_["regressor__n_estimators"],
        max_depth=random_search.best_params_["regressor__max_depth"],
        min_samples_split=random_search.best_params_["regressor__min_samples_split"],
        min_samples_leaf=random_search.best_params_["regressor__min_samples_leaf"],
        max_features=random_search.best_params_["regressor__max_features"],
        random_state=42,
    ))
])
final_model.fit(X, y)
model = final_model



# In[ ]:

model_path = Path("./models/rf_tuned_fst_model_imp_bundle.pkl")


# In[ ]:


model_bundle = {
    "model": model,
    "metrics": {
        "rmse": rmse,
    },
    "test_size": test_split_size,
    "best_params": random_search.best_params_,
    "param_grid": param_dist,
    "metadata": {
        "trained_on": str(datetime.datetime.now()),
        "model_type": "RandomForestRegressor",
        "features": list(X.columns),
        "target": "Listening_Time_minutes",
    },
}

joblib.dump(model_bundle, model_path)
