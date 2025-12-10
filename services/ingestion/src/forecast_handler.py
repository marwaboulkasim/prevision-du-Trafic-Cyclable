from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from db_handler import DBHandler


class ForecastHandler:
    def __init__(self) -> None:
        self.db_handler = DBHandler()
        self.df: pd.DataFrame = pd.DataFrame()

    def provide_forecast_features(self, best_counters_df):
        today = date.today()
        self.forecast_df = best_counters_df.copy()
        last_28_days_df = self.db_handler.select_last_28_days(
            self.forecast_df
        ).last_28_days_df.copy()  # pyright: ignore[reportAttributeAccessIssue]
        last_28_days_df["date"] = pd.to_datetime(last_28_days_df["date"])
        self.forecast_df["rounded_coordinates"] = self.forecast_df[
            "rounded_coordinates"
        ].apply(lambda x: (x[0], x[1]))
        self.forecast_df["date"] = pd.to_datetime(today)
        self.forecast_df["year"] = self.forecast_df["date"].dt.year
        self.forecast_df["month"] = self.forecast_df["date"].dt.month
        self.forecast_df["day"] = self.forecast_df["date"].dt.day
        self.forecast_df["weekday"] = self.forecast_df["date"].dt.day_of_week
        self.forecast_df["is_weekend"] = (
            self.forecast_df["weekday"].isin([5, 6]).astype(int)
        )

        today_ts = pd.Timestamp(today)

        def get_lag_7d(row):
            counter_data = last_28_days_df[
                last_28_days_df["counter_id"] == row["counter_id"]
            ]
            for offset in [0, 7, 14, 21]:
                target_date = today_ts - timedelta(days=7 + offset)
                match = counter_data[counter_data["date"] == target_date]
                if len(match) > 0:
                    return match.iloc[0]["intensity"]  # pyright: ignore[reportAttributeAccessIssue]
            return np.nan

        def get_lag_28d(row):
            counter_data = last_28_days_df[
                last_28_days_df["counter_id"] == row["counter_id"]
            ]
            for offset in [0, 7, 14, 21]:
                target_date = today_ts - timedelta(days=28 - offset)
                match = counter_data[counter_data["date"] == target_date]
                if len(match) > 0:
                    return match.iloc[0]["intensity"]  # pyright: ignore[reportAttributeAccessIssue]
            return np.nan

        def get_rolling(row, window_days):
            counter_data = last_28_days_df[
                last_28_days_df["counter_id"] == row["counter_id"]
            ]
            if window_days == 7:
                for offset in [0, 7, 14, 21]:
                    window_end = today_ts - timedelta(days=1 + offset)
                    window_start = window_end - timedelta(days=6)
                    window_data = counter_data[
                        (counter_data["date"] >= window_start)
                        & (counter_data["date"] <= window_end)
                    ]
                    if len(window_data) >= 4:
                        return round(window_data["intensity"].mean(), 2)
                return np.nan
            else:
                window_end = today_ts - timedelta(days=1)
                window_start = window_end - timedelta(days=27)
                window_data = counter_data[
                    (counter_data["date"] >= window_start)
                    & (counter_data["date"] <= window_end)
                ]
                if len(window_data) >= 14:
                    return round(window_data["intensity"].mean(), 2)
                return np.nan

        self.forecast_df["lag_7d"] = self.forecast_df.apply(
            lambda row: get_lag_7d(row), axis=1
        )
        self.forecast_df["lag_28d"] = self.forecast_df.apply(
            lambda row: get_lag_28d(row), axis=1
        )
        self.forecast_df["rolling_7d"] = self.forecast_df.apply(
            lambda row: get_rolling(row, 7), axis=1
        )
        self.forecast_df["rolling_28d"] = self.forecast_df.apply(
            lambda row: get_rolling(row, 28), axis=1
        )
        self.forecast_df["date"] = self.forecast_df["date"].astype(str)
        response_data = []
        for i in self.forecast_df["rounded_coordinates"].unique():
            latitude, longitude = i[0], i[1]
            try:
                response = requests.get(
                    f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=rain_sum,temperature_2m_mean&forecast_days=1"
                )
                response_data.append(
                    {
                        "rounded_coordinates": i,
                        "rain": response.json().get("daily", {}).get("rain_sum")[0],
                        "temperature": response.json()
                        .get("daily", {})
                        .get("temperature_2m_mean")[0],
                    }
                )
            except Exception as e:
                print(e)
                return e
        temp_df = pd.DataFrame(response_data)
        self.forecast_df = pd.merge(
            self.forecast_df, temp_df, how="left", on="rounded_coordinates"
        )
        self.forecast_df["is_rainy"] = self.forecast_df.apply(
            lambda row: 1 if row["rain"] >= 0.1 else 0, axis=1
        )

        return self
