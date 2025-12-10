import os
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TARGET_COLUMN = "intensity"


def load_data_from_supabase(table_name="historical_data") -> pd.DataFrame:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    response = supabase.table(table_name).select("*").execute()
    
    # Vérifier que la réponse contient bien des données
    if not hasattr(response, "data") or response.data is None:
        raise Exception("Erreur lors du chargement des données depuis Supabase")
    
    df = pd.DataFrame(response.data)
    
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"La colonne '{TARGET_COLUMN}' n'existe pas dans la table Supabase")
    
    return df
