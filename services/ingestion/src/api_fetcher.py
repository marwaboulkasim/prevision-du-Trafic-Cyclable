import pandas as pd
import requests


class APIFetcher:
    def __init__(self):
        self.counters_df: pd.DataFrame = pd.DataFrame()
        self.raw_df: pd.DataFrame = pd.DataFrame()

    def fetch_counters(self):
        """
        Fetch all the available counters and store their ids and coordinates in a dataframe
        """
        url = "https://portail-api-data.montpellier3m.fr/ecocounter?limit=1000"
        response = requests.get(url)
        self.counters_df = pd.json_normalize(response.json())
        self.counters_df = self.counters_df[["id", "location.value.coordinates"]]  # pyright: ignore[reportAttributeAccessIssue]
        self.counters_df = self.counters_df.rename(
            columns={"location.value.coordinates": "coordinates"}
        )
        return self

    def fetch_intensity(self):
        """
        Fetch intensity data for every year since made available and for every counter
        """
        years = ["2022", "2023", "2024", "2025"]
        response_data = []

        for id in self.counters_df["id"]:
            for year in years:
                from_date = f"{year}-01-01"
                to_date = f"{year}-12-01"
                response = requests.get(
                    f"https://portail-api-data.montpellier3m.fr/ecocounter_timeseries/{id}/attrs/intensity?fromDate={from_date}T00%3A00%3A00&toDate={to_date}T00%3A00%3A00"
                )
                response_data.append(response.json())

        id = [item.get("entityId", {}) for item in response_data]
        datetime = [item.get("index") for item in response_data]
        intensity = [item.get("values") for item in response_data]

        dfs = []

        for i in range(len(id)):
            temp_df = pd.DataFrame(
                {
                    "id": id[i],
                    "datetime": datetime[i],
                    "intensity": intensity[i],
                }
            )
            if not temp_df.empty:
                dfs.append(temp_df)

        self.df = pd.concat(dfs, ignore_index=True)
        return self
