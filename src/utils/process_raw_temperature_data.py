from datetime import datetime
import pathlib
import pandas as pd
from missing_data_detection import get_na_value_for_missing_data

cwd = pathlib.Path().resolve()
raw_path = f"{cwd}/data/raw/temperature"
df = pd.read_csv(f"{cwd}/data/raw/temperature/nobeltorget_malmo_4402725.csv", delimiter=";")
df.drop_duplicates(inplace=True)
df.rename(columns={"Datum": "date", "Klockslag": "time", "Timmedel": "temperature"}, inplace=True)

df["date"] = df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
df["hour"] = df["time"].apply(lambda x: datetime.strptime(x, "%H:%M:%S").hour)
df.drop(columns=["time"], inplace=True)
df = df[["date", "hour", "temperature"]]

# If some hours on some dates are missing, add na values for them so we can impute them later.
df_na_rows_for_dates_with_missing_data = get_na_value_for_missing_data(df)
print(
    f"Number of added na value rows due to missing hourly data (for later imputing): {len(df_na_rows_for_dates_with_missing_data)}"
)
df = pd.concat([df, df_na_rows_for_dates_with_missing_data])

df.sort_values(by=["date", "hour"], inplace=True, ignore_index=True)

temperature_num_na = int(df["temperature"].isna().sum())
print(f"Number of missing values in temperature column: {temperature_num_na}")
print(f"Percent missing values in temperature column: {temperature_num_na / len(df) * 100:.2f}%")
print("Imputing missing values with the previous value.")
df["temperature"] = df["temperature"].bfill()

# exclude duplicate data points. One source of duplicates is daylight savings.
# For example 2022-10-30 hour 2 in the raw data. Daylight savings time ends on this date.
# Sunday, 30 October 2022, 03:00:00 clocks were turned backward 1 hour to 02:00:00 instead.
print("Excluding day and hour duplicates (by keeping last)")
df = df.drop_duplicates(subset=["date", "hour"], keep="last")

output_path = f"{cwd}/data/processed/temperature.csv"
print("Saving processed temperature data to: ", output_path)
df.to_csv(output_path, index=False, sep=";")