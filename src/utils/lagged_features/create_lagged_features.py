import os
import shutil
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

df = pl.read_csv(
    "data/processed/day_ahead_price.csv", separator=";", decimal_comma=True
)

# Include features so that the time between the target price and each feature is at least 24 hours (the forecast horizon)
lagged_df = df.select(
    "price",
    "ds",
    *[
        pl.col("price").shift(i).alias(f"lagged_price_{i}h")
        for i in [24, 24 + 1, 24 + 2, 24 + 3]
    ],
    lagged_price_7d=pl.col("price").shift(7 * 24),
    lagged_price_24h_mean_24h=pl.col("price").shift(24).rolling_mean(24),
    lagged_price_24h_mean_7d=pl.col("price").shift(24).rolling_mean(7 * 24),
)
# drop the nulls that resulted from lagging
lagged_df = lagged_df.drop_nulls()

lagged_features_dir = "data/lagged_features"
if os.path.exists(lagged_features_dir):
    shutil.rmtree(lagged_features_dir)

os.makedirs(lagged_features_dir)
lagged_df.to_pandas().to_csv(
    f"{lagged_features_dir}/lagged_features.csv", index=False, sep=";"
)

max_train_size = 24 * 7 * 4  # one month of training data
test_size = 48  # two days of test
n_splits = len(lagged_df) // (max_train_size + test_size)

ts_cv = TimeSeriesSplit(
    n_splits=n_splits, max_train_size=max_train_size, test_size=test_size
)
all_splits = list(ts_cv.split(lagged_df))

base_dir = f"{lagged_features_dir}/time_series_split"
# remove base_dir and all its contents if it exists
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

os.makedirs(base_dir)

for idx, split in enumerate(all_splits):
    split_dir = f"{base_dir}/{idx}"
    os.makedirs(split_dir)

    train_idx, test_idx = split
    train = lagged_df[train_idx]
    test = lagged_df[test_idx]

    train_output_path = f"{split_dir}/train.csv"
    train.to_pandas().to_csv(train_output_path, index=False, sep=";")

    test_output_path = f"{split_dir}/test.csv"
    test.to_pandas().to_csv(test_output_path, index=False, sep=";")