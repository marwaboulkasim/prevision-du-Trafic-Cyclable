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
        self.weather_df["date"] = self.weather_df["datetime"].dt.date  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
        self.weather_df["date"] = pd.to_datetime(self.df["date"])  # pyright: ignore[reportIndexIssue]

        self.weather_df = self.weather_df.groupby(
            ["rounded_coordinates", "date"], as_index=False
        ).agg({"temperature": "mean", "rain": "sum"})

        self.df = pd.merge(
            self.df,
            self.weather_df,  # pyright: ignore[reportArgumentType]
            on=["rounded_coordinates", "date"],
            how="left",
        )
        self.df["is_rainy"] = self.df.apply(
            lambda row: 1 if row["rain"] >= 0.1 else 0, axis=1
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
        self.df = self.df.drop_duplicates(
            subset=["counter_id", "datetime"], ignore_index=True
        )
        return self

    def add_features(self):
        """
        Use existing columns to create new features.
        """
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day
        self.df["weekday"] = self.df["date"].dt.day_of_week
        self.df["rolling_7d"] = round(
            self.df.groupby("counter_id")["intensity"]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True),
            2,
        )
        self.df["rolling_28d"] = round(
            self.df.groupby("counter_id")["intensity"]
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True),
            2,
        )
        self.df["lag_7d"] = self.df.groupby("counter_id")["intensity"].shift(7)
        self.df["lag_7d"] = self.df["lag_7d"].fillna(self.df["rolling_7d"]).astype(int)
        self.df["lag_28d"] = self.df.groupby("counter_id")["intensity"].shift(28)
        self.df["lag_28d"] = (
            self.df["lag_28d"].fillna(self.df["rolling_28d"]).astype(int)
        )
        self.df["is_weekend"] = self.df.apply(
            lambda row: 1 if row["weekday"] in [5, 6] else 0, axis=1
        )
        return self

    def convert_to_daily_values(self):
        self.df["date"] = self.df["datetime"].dt.date
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.groupby(  # pyright: ignore[reportAttributeAccessIssue]
            by=["counter_id", "date", "coordinates", "rounded_coordinates"],
            as_index=False,
        ).agg(
            {
                "intensity": "sum",
            }
        )
        if "datetime" in self.df.columns:
            self.df = self.df.drop(columns="datetime")
        return self

    def convert_date_to_string(self):
        """
        Convert 'date' column to string before database insertion.
        """
        self.df["date"] = self.df["date"].astype(str)
        return self

    def clean(self):
        """ """
        self.df["temperature"] = self.df["temperature"].ffill()
        self.df["rain"] = self.df["rain"].fillna(0)
        self.df = self.df[  # pyright: ignore[reportAttributeAccessIssue]
            [
                "counter_id",
                "coordinates",
                "rounded_coordinates",
                "date",
                "year",
                "month",
                "day",
                "weekday",
                "is_weekend",
                "rolling_7d",
                "rolling_28d",
                "lag_7d",
                "lag_28d",
                "temperature",
                "rain",
                "is_rainy",
                "intensity",
            ]
        ]
        return self
