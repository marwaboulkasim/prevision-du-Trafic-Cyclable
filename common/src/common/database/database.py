import os

from dotenv import load_dotenv
from supabase import create_client

_ = load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    raise ValueError("SUPABASE_URL and/or SUPABASE_KEY not found")
