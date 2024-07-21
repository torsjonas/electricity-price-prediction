#import csv
import os
import pathlib
import pandas as pd
from bs4 import BeautifulSoup

def file_to_pandas_dataframe(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as f:
        soup = BeautifulSoup(f.read(), features="html.parser")
        header_row = soup.select_one("tr.column-headers")
        data_rows = soup.select("tr.data-row")

        headers = [th.text for th in header_row.find_all("th")]
        rows = [[td.text for td in row.find_all("td")] for row in data_rows]

        df = pd.DataFrame(rows, columns=headers)
        df.rename(columns={df.columns[0]: "hour interval"}, inplace=True)
        df = df.drop(columns=["hour interval"])
        df = df.reset_index()
        df = df.rename(columns={"index": "hour"})
        df = df[df["hour"] < 24]
        # Each date is the name of a column. Pivot so that date is instead a value in a "date" column.
        df = df.melt(id_vars=["hour"], var_name="date", value_name="price")

        return df

cwd = pathlib.Path().resolve()
raw_path = f"{cwd}/data/raw"
items = os.listdir(raw_path)
files = [f"{raw_path}/{item}" for item in items if item.endswith(".xls")]

dfs = [file_to_pandas_dataframe(file) for file in files]
df = pd.concat(dfs)
# drop the inevitable duplicates from manual clicking on the website to download the files
# we may also be missing some data, but we shall see how robust the model is to that.

df.drop_duplicates(inplace=True)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# in case the data is not sorted after merging files, sort it by date and hour
df.sort_values(by=["date", "hour"], inplace=True, ignore_index=True)

df.to_csv(f"{cwd}/data/processed/processed_day_ahead_prices.csv", index=False)

#df_read = pd.read_csv(f"{cwd}/data/processed/processed_day_ahead_prices.csv", decimal=",")
#print(df_read.head())