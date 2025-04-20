#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

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


train_df = load_data("train.csv")


# In[ ]:


X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_size, random_state=42
)


# In[2]:


param_dist = {
    "n_estimators": [160, 180, 200, 220, 240],
    "max_depth": [None, 26, 28, 30, 32, 34],
    "min_samples_split": [2, 3],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}

rf = RandomForestRegressor(random_state=42)

cv = KFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    verbose=1,
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
final_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
final_model.fit(X, y)
model = final_model


# In[ ]:


importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))


# In[ ]:


model_path = Path("./models/rf_tuned_thrd_model_bundle.pkl")


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
