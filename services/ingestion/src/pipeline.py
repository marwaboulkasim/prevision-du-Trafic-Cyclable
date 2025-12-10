from typing import final

from api_fetcher import APIFetcher
from data_transformer import DataTransformer
from db_handler import DBHandler
from forecast_handler import ForecastHandler


@final
class IngestionPipeline:
    def __init__(self):
        self.api_fetcher = APIFetcher()
        self.data_transformer = DataTransformer()
        self.db_handler = DBHandler()
        self.forecast_handler = ForecastHandler()

    def run(self):
        is_table_filled = self.db_handler.check_content()
        if is_table_filled:
            print("Inserting daily forecast data")
            self.forecast_handler.provide_forecast_features(
                self.db_handler.select_best_counters().best_counters_df  # pyright: ignore[reportAttributeAccessIssue]
            )
            _ = self.db_handler.insert_forecast_data(
                self.forecast_handler.forecast_df.to_dict(orient="records")
            )
        else:
            print("Fetching data...")
            _ = (
                self.api_fetcher.fetch_counters()
                .fetch_historical_data()
                .fetch_weather_data()
            )
            print("Successfully fetched data")

            print("Applying transformation methods to data...")
            self.api_fetcher.historical_data = (
                self.data_transformer.load_historical_df(
                    self.api_fetcher.historical_data
                )
                .load_counters_df(self.api_fetcher.counters_df)
                .add_coordinates()
                .apply_basic_transformations()
                .convert_to_daily_values()
                .add_features()
                .load_weather_df(self.api_fetcher.weather_data)
                .add_weather()
                .clean()
                .keep_top_counters()
                .convert_date_to_string()
                .df
            )
            print("Successfully transformed data")
            print(self.api_fetcher.historical_data)

            print("Inserting best counters in db...")
            _ = self.db_handler.insert_best_counters(
                self.data_transformer.best_counters.to_dict(orient="records")
            )

            # print("Fetching new historical data...")
            # _ = self.api_fetcher.fetch_counters().fetch_new_historical_data()
            # print(self.api_fetcher.new_historical_data)
            # print("Successfully fetched new historical data")

            # _ = self.db_handler.insert(
            #     self.api_fetcher.historical_data.to_dict(orient="records")
            # )
