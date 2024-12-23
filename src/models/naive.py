import numpy as np
class Naive:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass 

    def predict(self, X: np.ndarray) -> np.ndarray:
        # For the naive forecast we predict the 24 lagged value, which is the first feature (column) of X.
        return X[:, 0]