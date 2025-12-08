from typing import final

from common.database.database import supabase  # pyright: ignore[reportMissingTypeStubs]


@final
class DBInserter:
    def __init__(self):
        self.client = supabase

    def insert(self, records, batch_size: int = 30000):
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                print(f"Inserting... : {i}/{len(records)}")
                _ = self.client.table("historical_data").insert(batch).execute()
        except Exception as e:
            return e
        print(f"Successfully inserted : {len(records)}/{len(records)}")
