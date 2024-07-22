from datetime import datetime
import os
import pathlib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from missing_data_detection import get_na_value_for_missing_data


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
raw_path = f"{cwd}/data/raw/day_ahead_prices"
items = os.listdir(raw_path)
files = [f"{raw_path}/{item}" for item in items if item.endswith(".xls")]

dfs = [file_to_pandas_dataframe(file) for file in files]
df = pd.concat(dfs)
# drop the inevitable duplicates from manual clicking on the website to download the files
# we may also be missing some data, but we shall see how robust the model is to that.
df.drop_duplicates(inplace=True)

df["date"] = df["date"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y").date())
df = df[["date", "hour", "price"]]

df_na_rows_for_dates_with_missing_data = get_na_value_for_missing_data(df)
print(
    f"Number of added na value rows due to missing hourly data (for later imputing): {len(df_na_rows_for_dates_with_missing_data)}"
)
df = pd.concat([df, df_na_rows_for_dates_with_missing_data])

df.sort_values(by=["date", "hour"], inplace=True, ignore_index=True)

df["price"] = df["price"].replace("", np.nan)

price_num_na = int(df["price"].isna().sum())
print(f"Number of missing values in price column: {price_num_na}")
print(f"Percent missing values in temperature column: {price_num_na / len(df) * 100:.2f}%")
print("Imputing missing values with the previous value.")
# impute missing values with the previous value (there are 16 missing values,
# so lets assume its not a big issue)
df["price"] = df["price"].bfill()

df.to_csv(f"{cwd}/data/processed/day_ahead_price.csv", index=False)
