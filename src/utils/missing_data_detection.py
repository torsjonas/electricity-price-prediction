from datetime import timedelta
import pandas as pd


def get_missing_hours(hours):
    expected_hours = list(range(24))
    for hour in hours:
        if hour in expected_hours:
            expected_hours.remove(hour)

    return expected_hours


def get_na_value_for_missing_data(df):
    dates = list(sorted(df["date"].unique()))

    na_value_rows = []
    for idx, curr_date in enumerate(dates):
        if idx < len(dates) - 1:
            next_date = dates[idx + 1]
            num_days_diff = (next_date - curr_date).days
            if num_days_diff > 1:
                for i in range(1, num_days_diff):
                    missing_date = curr_date + timedelta(days=i)
                    missing_hours = get_missing_hours([])
                    for hour in missing_hours:
                        na_value_rows.append([missing_date, hour, None])
            else:
                missing_hours = get_missing_hours(
                    df[df["date"] == curr_date]["hour"].values.tolist()
                )
                for hour in missing_hours:
                    na_value_rows.append([curr_date, hour, None])
    
    na_values_df = pd.DataFrame(na_value_rows, columns=df.columns)
    na_values_df["hour"].astype(int)

    return na_values_df
