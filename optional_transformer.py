# optional_transformer.py

from sklearn.base import BaseEstimator, TransformerMixin


class OptionalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None):
        self.transformer = transformer

    def fit(self, X, y=None):
        if self.transformer is not None:
            self.transformer.fit(X, y)
        return self

    def transform(self, X):
        if self.transformer is not None:
            return self.transformer.transform(X)
        return X
