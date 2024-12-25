import pandas as pd
class Naive:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass 

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # For the naive forecast we predict the 24 lagged value, which is the first feature (column) of X.
        return X["lag_24h"].to_frame("y")
