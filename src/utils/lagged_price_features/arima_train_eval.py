import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from statsforecast import StatsForecast

def get_sorted_split_dirs(time_series_split_dir):
    split_dirs = [
        os.path.join(time_series_split_dir, split_dir)
        for split_dir in os.listdir(time_series_split_dir)
    ]
    split_dirs = sorted(split_dirs, key=lambda x: int(x.split("/")[-1]))
    return split_dirs


def train_and_predict(
    sf: StatsForecast, train: pd.DataFrame, test: pd.DataFrame, exogenous_features=None
):
    h = len(test)
    train_cols = ["y", "ds", "unique_id"]
    train_cols = train_cols + exogenous_features if exogenous_features else train_cols

    df = train[train_cols].copy()
    df["ds"] = df["ds"].apply(pd.Timestamp)

    X_df = None
    if exogenous_features:
        X_df = test[["ds", "unique_id"] + exogenous_features].copy()
        X_df["ds"] = X_df["ds"].apply(pd.Timestamp)

    pred = sf.forecast(df=df, h=h, X_df=X_df, level=[90])
    pred["y"] = pred["AutoARIMA"]

    return pred


def train_eval_mape(sf: StatsForecast, train, test):
    pred = train_and_predict(sf=sf, train=train, test=test)
    mape = mean_absolute_percentage_error(test["y"], pred["y"])
    return mape


def create_train_test_df_from_split_dir(split_dir):
    train = pd.read_csv(os.path.join(split_dir, "train.csv"), delimiter=";", header=0)
    test = pd.read_csv(os.path.join(split_dir, "test.csv"), delimiter=";", header=0)

    return train, test


def train_plot_preds_from_split_dir(
    sf: StatsForecast, split_dir, exogenous_features=None
):
    train, test = create_train_test_df_from_split_dir(split_dir)
    pred = train_and_predict(sf=sf, train=train, test=test, exogenous_features=exogenous_features)

    df_y_train = pd.DataFrame(train, columns=["y"])
    df_y_train["type"] = "truth"
    df_y_train.reset_index(inplace=True)

    df_y_test = pd.DataFrame(test, columns=["y"])
    df_y_test["type"] = "truth"
    df_y_test.reset_index(inplace=True)

    df_y_pred = pd.DataFrame(pred, columns=["y"])
    df_y_pred["type"] = "pred"
    df_y_pred.reset_index(inplace=True)

    df_y_test["index_offset"] = len(df_y_train)
    df_y_test["index"] = df_y_test["index"] + df_y_test["index_offset"] + 1

    df_y_pred["index_offset"] = len(df_y_train)
    df_y_pred["index"] = df_y_pred["index"] + df_y_pred["index_offset"] + 1

    df_all = pd.concat([df_y_train, df_y_test, df_y_pred], ignore_index=True)
    df_all = df_all.sort_values(by="index")
    sns.lineplot(data=df_all, x="index", y="y", hue="type")

    return pred


def train_eval_mape_from_split_dir(sf: StatsForecast, split_dir):
    print(f"Evaluating split {split_dir}")
    train, test = create_train_test_df_from_split_dir(split_dir)
    return train_eval_mape(sf=sf, train=train, test=test)


def calculate_mape(sf: StatsForecast, split_dirs):
    return [
        train_eval_mape_from_split_dir(sf=sf, split_dir=split_dir) for split_dir in split_dirs
    ]


def plot_mape_for_split_indices(split_dirs, mape):
    split_indices = [split_dir.split("/")[-1] for split_dir in split_dirs]

    # set the title of the plot
    sns.barplot(x=split_indices, y=mape)
    plt.title("MAPE for each split")
