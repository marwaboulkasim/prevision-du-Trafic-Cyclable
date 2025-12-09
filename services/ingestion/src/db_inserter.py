import sys
from typing import final

from common.database.database import supabase  # pyright: ignore[reportMissingTypeStubs]


@final
class DBInserter:
    def __init__(self):
        self.client = supabase

    def insert(self, records, batch_size: int = 5000):
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                sys.stdout.write(
                    f"\rInserting... : {min(i + batch_size, len(records))}/{len(records)}"
                )
                sys.stdout.flush()
                _ = self.client.table("historical_data_test").insert(batch).execute()

            sys.stdout.write("\n")
        except Exception as e:
            print(e)
            return e
        print(f"Successfully inserted : {len(records)}/{len(records)}")
