import datetime
import sys
from typing import final

import pandas as pd
from common.database.database import supabase  # pyright: ignore[reportMissingTypeStubs]


@final
class DBHandler:
    def __init__(self) -> None:
        self.client = supabase
        self.historical_table: str = "historical_data"
        self.forecast_table: str = "forecast_data"
        self.best_counters_table: str = "best_counters"
        self.last_28_days_df: pd.DataFrame = pd.DataFrame()

    def insert(self, records, batch_size: int = 5000):
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                sys.stdout.write(
                    f"\rInserting... : {min(i + batch_size, len(records))}/{len(records)}"
                )
                sys.stdout.flush()
                _ = self.client.table(self.historical_table).insert(batch).execute()

            sys.stdout.write("\n")
        except Exception as e:
            return e
        print(f"Successfully inserted : {len(records)}/{len(records)}")

    def check_content(self):
        try:
            response = (
                self.client.table(self.historical_table).select("*").limit(10).execute()
            )
            is_table_filled = response.data != []
            return is_table_filled
        except Exception as e:
            return e

    def select_last_28_days(self, best_counters_df: pd.DataFrame):
        date_28_days_ago = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(28)
        ).strftime("%Y-%m-%d")
        try:
            response = (
                self.client.table(self.historical_table)
                .select("*")
                .in_("counter_id", best_counters_df["counter_id"].tolist())
                .gte("date", date_28_days_ago)
                .execute()
            )
            self.last_28_days_df = pd.DataFrame.from_records(response.data)
            return self
        except Exception as e:
            return e

    def insert_best_counters(self, records):
        try:
            _ = self.client.table(self.best_counters_table).insert(records).execute()
        except Exception as e:
            print(e)
            return e
        print(
            f"Successfully inserted best counters in {self.best_counters_table} table"
        )

    def select_best_counters(self):
        try:
            response = self.client.table(self.best_counters_table).select("*").execute()
            self.best_counters_df: pd.DataFrame = pd.DataFrame.from_records(
                response.data
            )
        except Exception as e:
            print(e)
            return e
        return self

    def insert_forecast_data(self, records):
        try:
            _ = self.client.table(self.forecast_table).insert(records).execute()
            print("Successfully inserted forecast data")
        except Exception as e:
            print(e)
            return e
        return self
