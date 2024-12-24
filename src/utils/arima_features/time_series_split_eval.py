import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error

def get_sorted_split_dirs(time_series_split_dir):
    split_dirs = [
        os.path.join(time_series_split_dir, split_dir)
        for split_dir in os.listdir(time_series_split_dir)
    ]
    split_dirs = sorted(split_dirs, key=lambda x: int(x.split("/")[-1]))
    return split_dirs

def train_and_predict(model, train: pd.DataFrame, test: pd.DataFrame, target_col="y"):
    model.fit(train)
    pred = model.predict(h=len(test), level=[90])
    pred[target_col] = pred["AutoARIMA"]

    return pred

def train_eval_mape(model, train, test, target_col="y"):
    pred = train_and_predict(model=model, train=train, test=test, target_col=target_col)
    mape = mean_absolute_percentage_error(test[target_col], pred[target_col])
    return mape

def create_train_test_df_from_split_dir(split_dir, target_col="y"):
    train = pd.read_csv(os.path.join(split_dir, "train.csv"), delimiter=";", header=0)
    test = pd.read_csv(os.path.join(split_dir, "test.csv"), delimiter=";", header=0)

    return train, test

def train_plot_preds_from_split_dir(model, split_dir, target_col="y"):
    train, test = create_train_test_df_from_split_dir(split_dir)
    pred = train_and_predict(model=model, train=train, test=test, target_col=target_col)

    df_y_train = pd.DataFrame(train, columns=[target_col])
    df_y_train["type"] = "truth"
    df_y_train.reset_index(inplace=True)

    df_y_test = pd.DataFrame(test, columns=[target_col])
    df_y_test["type"] = "truth"
    df_y_test.reset_index(inplace=True)

    df_y_pred = pd.DataFrame(pred, columns=[target_col])
    df_y_pred["type"] = "pred"
    df_y_pred.reset_index(inplace=True)

    df_y_test["index_offset"] = len(df_y_train)
    df_y_test["index"] = df_y_test["index"] + df_y_test["index_offset"] + 1

    df_y_pred["index_offset"] = len(df_y_train)
    df_y_pred["index"] = df_y_pred["index"] + df_y_pred["index_offset"] + 1

    df_all = pd.concat([df_y_train, df_y_test, df_y_pred], ignore_index=True)
    df_all = df_all.sort_values(by="index")
    sns.lineplot(data=df_all, x="index", y=target_col, hue="type")

    return pred

def train_eval_mape_from_split_dir(model, split_dir):
    print(f"Evaluating split {split_dir}")
    train, test = create_train_test_df_from_split_dir(split_dir)
    return train_eval_mape(model=model, train=train, test=test)

def calculate_mape(model, split_dirs):
    return [train_eval_mape_from_split_dir(model, split_dir) for split_dir in split_dirs]

def plot_mape_for_split_indices(split_dirs, mape):
    split_indices = [split_dir.split("/")[-1] for split_dir in split_dirs]

    # set the title of the plot
    sns.barplot(x=split_indices, y=mape)
    plt.title("MAPE for each split")

