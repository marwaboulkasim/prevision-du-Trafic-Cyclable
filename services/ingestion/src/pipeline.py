from typing import final

from api_fetcher import APIFetcher
from data_transformer import DataTransformer
from db_inserter import DBInserter


@final
class IngestionPipeline:
    def __init__(self):
        self.api_fetcher = APIFetcher()
        self.data_transformer = DataTransformer()
        self.db_inserter = DBInserter()

    def run(self):
        print("Fetching data...")
        _ = self.api_fetcher.fetch_counters().fetch_historical_data()
        print("Successfully fetched data")

        print("Applying transformation methods to data...")
        self.api_fetcher.historical_data = (
            self.data_transformer.apply_basic_transformations(
                self.api_fetcher.historical_data
            )
        )
        self.api_fetcher.historical_data = self.data_transformer.add_features(
            self.api_fetcher.historical_data
        )
        self.api_fetcher.historical_data = (
            self.data_transformer.convert_datetime_to_string(
                self.api_fetcher.historical_data
            )
        )
        self.api_fetcher.historical_data = self.data_transformer.clean(
            self.api_fetcher.historical_data
        )
        print("Successfully transformed data")

        # print("Fetching new historical data...")
        # _ = self.api_fetcher.fetch_new_historical_data()
        # print(self.api_fetcher.new_historical_data)
        # print("Successfully fetched new historical data")

        # _ = self.db_inserter.insert(
        #     self.api_fetcher.historical_data.to_dict(orient="records")
        # )
