import pandas as pd


class DataTransformer:
    def apply_basic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert 'datetime' column to datetime to be able to use what this data type provides within pandas.
        Rename 'id' column.
        Convert 'intensity' column to int type.
        """
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"id": "counter_id"})
        df["intensity"] = df["intensity"].astype(int)
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use existing columns to create new features.
        """
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["hour"] = df["datetime"].dt.hour
        df["weekday"] = df["datetime"].dt.day_of_week
        df["rolling_7d"] = round(
            df.groupby("counter_id")["intensity"]
            .rolling(window=24 * 7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True),
            2,
        )
        df["rolling_28d"] = round(
            df.groupby("counter_id")["intensity"]
            .rolling(24 * 28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True),
            2,
        )
        df["lag_7d"] = df.groupby("counter_id")["intensity"].shift(24 * 7)
        df["lag_7d"] = df["lag_7d"].fillna(df["rolling_7d"]).astype(int)
        df["lag_28d"] = df.groupby("counter_id")["intensity"].shift(24 * 28)
        df["lag_28d"] = df["lag_28d"].fillna(df["rolling_28d"]).astype(int)
        df["is_weekend"] = df.apply(
            lambda row: 1 if row["weekday"] in [5, 6] else 0, axis=1
        )
        df = df[  # pyright: ignore[reportAssignmentType]
            [
                "counter_id",
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
                "intensity",
            ]
        ]
        return df

    def convert_datetime_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert 'datetime' column to string before database insertion.
        """
        df["datetime"] = (
            df["datetime"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """ """
        df = df.drop_duplicates(subset=["counter_id", "datetime"])
        return df
