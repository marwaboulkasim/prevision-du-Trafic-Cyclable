import time
from datetime import date, timedelta

import pandas as pd
import requests


class APIFetcher:
    def __init__(self):
        self.counters_df: pd.DataFrame = pd.DataFrame()
        self.historical_data: pd.DataFrame = pd.DataFrame()
        self.new_historical_data: pd.DataFrame = pd.DataFrame()
        self.weather_data: pd.DataFrame = pd.DataFrame()

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
        self.counters_df["coordinates"] = self.counters_df["coordinates"].apply(
            lambda x: (x[0], x[1])
        )
        self.counters_df["rounded_coordinates"] = self.counters_df["coordinates"].apply(
            lambda x: (round(x[0], 2), round(x[1], 2))
        )
        return self

    def fetch_historical_data(self):
        """
        Fetch historical data for every year since made available and for every counter
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

        self.historical_data = pd.concat(dfs, ignore_index=True)
        return self

    def fetch_new_historical_data(self):
        """
        Fetch yesterday's historical data
        """
        today = date.today()
        last_date_in_db = "2025-11-30"  # temporaire

        response_data = []

        for id in self.counters_df["id"]:
            response = requests.get(
                f"https://portail-api-data.montpellier3m.fr/ecocounter_timeseries/{id}/attrs/intensity?fromDate={str(last_date_in_db)}T00%3A00%3A00&toDate={str(today)}T00%3A00%3A00"
            )
            print(response.json())
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

        self.new_historical_data = pd.concat(dfs, ignore_index=True)
        return self

    def fetch_weather_data(self):
        """
        Fetch hourly weather data for every counter's location
        """
        start_date = "2022-01-01"
        end_date = date.today() - timedelta(1)

        response_data = []
        known_coordinates = []

        loop = 0
        max_calls_per_minute = 25
        call_count = 0
        start_time = time.time()

        for id in self.counters_df["id"]:
            print(f"\nLOOP {loop}")
            latitude, longitude = self.counters_df.loc[
                self.counters_df["id"] == id, "rounded_coordinates"
            ].values[0]

            if [latitude, longitude] not in known_coordinates:
                known_coordinates.append([latitude, longitude])

                if call_count >= max_calls_per_minute:
                    elapsed = time.time() - start_time
                    if elapsed < 60:
                        sleep_time = 60 - elapsed
                        print(f" Pausing for {int(sleep_time)} seconds (API limit)")
                        time.sleep(sleep_time)
                    call_count = 0
                    start_time = time.time()

                print(
                    f"API CALL ({call_count}) FOR {id} LOCATED AT ({latitude}, {longitude})"
                )
                response = requests.get(
                    f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,rain"
                )
                print(f"{response.json()}"[:500])
                response_data.append(
                    {
                        "rounded_coordinates": (latitude, longitude),
                        "response": response.json(),
                    }
                )
                call_count += 1
            else:
                print("Coordinates have already been covered")
            loop += 1

        rounded_coordinates = [
            item.get("rounded_coordinates", {}) for item in response_data
        ]
        datetime = [
            item.get("response", {}).get("hourly", {}).get("time", {})
            for item in response_data
        ]
        temperature = [
            item.get("response", {}).get("hourly", {}).get("temperature_2m", {})
            for item in response_data
        ]
        rain = [
            item.get("response", {}).get("hourly", {}).get("rain", {})
            for item in response_data
        ]

        self.weather_data = pd.DataFrame(
            {
                "rounded_coordinates": rounded_coordinates,
                "datetime": datetime,
                "temperature": temperature,
                "rain": rain,
            }
        )
        self.weather_data = self.weather_data.explode(
            ["datetime", "temperature", "rain"], ignore_index=True
        )
        self.weather_data["datetime"] = pd.to_datetime(
            self.weather_data["datetime"], utc=True
        )
        print(self.weather_data)
        return self
