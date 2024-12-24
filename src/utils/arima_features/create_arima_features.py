import os
import shutil
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

df = pl.read_csv(
    "data/processed/day_ahead_price.csv", separator=";", decimal_comma=True
)

arima_df = df.select(
    pl.col("price").alias("y"),
    "ds",
).with_columns(pl.lit("day_ahead_price").alias("unique_id"))

arima_features_dir = "data/arima_features"
if os.path.exists(arima_features_dir):
    shutil.rmtree(arima_features_dir)

os.makedirs(arima_features_dir)
arima_df.to_pandas().to_csv(
    f"{arima_features_dir}/arima_features.csv", index=False, sep=";"
)

max_train_size = 24 * 7 * 4  # one month of training data
test_size = 48  # two days test data
n_splits = len(arima_df) // (max_train_size + test_size)

ts_cv = TimeSeriesSplit(
    n_splits=n_splits, max_train_size=max_train_size, test_size=test_size
)
all_splits = list(ts_cv.split(arima_df))

base_dir = f"{arima_features_dir}/time_series_split"
# remove base_dir and all its contents if it exists
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

os.makedirs(base_dir)

for idx, split in enumerate(all_splits):
    split_dir = f"{base_dir}/{idx}"
    os.makedirs(split_dir)

    train_idx, test_idx = split
    train = arima_df[train_idx]
    test = arima_df[test_idx]

    train_output_path = f"{split_dir}/train.csv"
    train.to_pandas().to_csv(train_output_path, index=False, sep=";")

    test_output_path = f"{split_dir}/test.csv"
    test.to_pandas().to_csv(test_output_path, index=False, sep=";")