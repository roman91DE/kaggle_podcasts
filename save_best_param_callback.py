import json
from sklearn_genetic.callbacks.base import BaseCallback


class SaveBestParamsCallback(BaseCallback):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def __call__(self, ga_search):
        try:
            best_params = ga_search.best_params_
            with open(self.filepath, "w") as f:
                json.dump(best_params, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save best parameters: {e}")