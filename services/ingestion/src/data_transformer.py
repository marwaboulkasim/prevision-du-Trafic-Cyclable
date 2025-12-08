import pandas as pd


class DataTransformer:
    def load_historical_df(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        return self

    def load_counters_df(self, df: pd.DataFrame):
        self.counters_df = df
        return self

    def load_weather_df(self, df: pd.DataFrame):
        self.weather_df = df
        return self

    def add_coordinates(self):
        if "coordinates" not in self.df.columns:
            self.df = pd.merge(self.df, self.counters_df, on="id")
        return self

    def add_weather(self):
        self.df = pd.merge(
            self.df, self.weather_df, on=["rounded_coordinates", "datetime"], how="left"
        )
        return self

    def apply_basic_transformations(self):
        """
        Convert 'datetime' column to datetime to be able to use what this data type provides within pandas.
        Rename 'id' column.
        Convert 'intensity' column to int type.
        """
        self.df["datetime"] = pd.to_datetime(self.df["datetime"], utc=True)
        self.df = self.df.rename(columns={"id": "counter_id"})
        self.df["intensity"] = self.df["intensity"].astype(int)
        return self

    def add_features(self):
        """
        Use existing columns to create new features.
        """
        self.df["year"] = self.df["datetime"].dt.year
        self.df["month"] = self.df["datetime"].dt.month
        self.df["day"] = self.df["datetime"].dt.day
        self.df["hour"] = self.df["datetime"].dt.hour
        self.df["weekday"] = self.df["datetime"].dt.day_of_week
        self.df["rolling_7d"] = round(
            self.df.groupby("counter_id")["intensity"]
            .rolling(window=24 * 7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True),
            2,
        )
        self.df["rolling_28d"] = round(
            self.df.groupby("counter_id")["intensity"]
            .rolling(24 * 28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True),
            2,
        )
        self.df["lag_7d"] = self.df.groupby("counter_id")["intensity"].shift(24 * 7)
        self.df["lag_7d"] = self.df["lag_7d"].fillna(self.df["rolling_7d"]).astype(int)
        self.df["lag_28d"] = self.df.groupby("counter_id")["intensity"].shift(24 * 28)
        self.df["lag_28d"] = (
            self.df["lag_28d"].fillna(self.df["rolling_28d"]).astype(int)
        )
        self.df["is_weekend"] = self.df.apply(
            lambda row: 1 if row["weekday"] in [5, 6] else 0, axis=1
        )
        self.df = self.df[  # pyright: ignore[reportAttributeAccessIssue]
            [
                "counter_id",
                "coordinates",
                "rounded_coordinates",
                "datetime",
                "year",
                "month",
                "day",
                "hour",
                "weekday",
                "is_weekend",
                "rolling_7d",
                "rolling_28d",
                "lag_7d",
                "lag_28d",
                "temperature",
                "rain",
                "intensity",
            ]
        ]
        return self

    def convert_datetime_to_string(self):
        """
        Convert 'datetime' column to string before database insertion.
        """
        self.df["datetime"] = (
            self.df["datetime"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        return self

    def clean(self):
        """ """
        self.df = self.df.drop_duplicates(
            subset=["counter_id", "datetime"], ignore_index=True
        )
        return self
