# services/frontend/app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import pydeck as pdk
from datetime import datetime

# -----------------------
# Config
# -----------------------
st.set_page_config(
    page_title="Prévision du Trafic Cyclable Montpellier",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_CSV = "../training/pipeline/data/raw/df_brut.csv"
MODEL_PKL = "../training/models/xgb_model.pkl"
METRICS_JSON = "../training/models/metrics.json"

# Features order expected by the model (adapter si besoin)
MODEL_FEATURES_ORDER = ["year", "month", "day", "hour", "weekday"]


# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_data(csv_path=DATA_CSV):
    # Lecture robuste du CSV
    try:
        df = pd.read_csv(csv_path, parse_dates=["time"])
    except ValueError:
        # si parse_dates provoque une erreur (colonne différente), on essaye une lecture simple puis conversion
        df = pd.read_csv(csv_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            raise FileNotFoundError("Le CSV doit contenir une colonne 'time' avec des timestamps.")
    # Vérification colonnes attendues
    for c in ["time", "value", "id", "coordinates"]:
        if c not in df.columns:
            raise KeyError(f"Colonne manquante dans le CSV : '{c}'")

    # Extraire features temporelles
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday  # 0 = Monday

    # Nettoyage / conversion coordinates -> lat / lon
    # Supporte formats : "(lat, lon)" ou "lat, lon" ou "[lat, lon]"
    def parse_coord(x):
        if pd.isna(x):
            return (np.nan, np.nan)
        s = str(x).strip()
        for ch in "()[]":
            s = s.replace(ch, "")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                # if they were reversed earlier (some files have (lon, lat)), detect unrealistic latitudes
                if abs(lat) > 90:  # then maybe order is (lon, lat)
                    lat, lon = lon, lat
                return (lat, lon)
            except Exception:
                return (np.nan, np.nan)
        return (np.nan, np.nan)

    coords = df["coordinates"].apply(parse_coord)
    df["lat"] = coords.apply(lambda t: t[0])
    df["lon"] = coords.apply(lambda t: t[1])

    return df


@st.cache_resource
def load_model_and_metrics(model_path=MODEL_PKL, metrics_path=METRICS_JSON):
    model = None
    metrics = {}
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.warning("Modèle introuvable — place ton .pkl dans ../training/models/ si tu veux des prédictions.")
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}
    return model, metrics


def build_input_df(year, month, day, hour, weekday, feature_order=MODEL_FEATURES_ORDER):
    row = {"year": int(year), "month": int(month), "day": int(day), "hour": int(hour), "weekday": int(weekday)}
    # Respecter l'ordre des features attendu par le modèle
    X = pd.DataFrame([row])
    # Keep only columns in feature_order and in X
    cols = [c for c in feature_order if c in X.columns]
    return X[cols]




st.title("Prévision du Trafic Cyclable — Montpellier")
st.markdown("Interface simple pour explorer les compteurs et tester le modèle.")

# Load data + model
try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

model, metrics = load_model_and_metrics()

# Sidebar controls
st.sidebar.header("Filtres & Prédiction rapide")
unique_weekdays = sorted(df["weekday"].dropna().unique().astype(int).tolist())
unique_months = sorted(df["month"].dropna().unique().astype(int).tolist())
unique_hours = sorted(df["hour"].dropna().unique().astype(int).tolist())
unique_ids = sorted(df["id"].dropna().unique().tolist())

jour = st.sidebar.selectbox("Jour de la semaine (0=Lundi)", unique_weekdays, index=0)
mois = st.sidebar.selectbox("Mois", unique_months, index=0)
heure = st.sidebar.selectbox("Heure", unique_hours, index=0)
station_sel = st.sidebar.selectbox("Compteur (id)", ["Tous"] + unique_ids, index=0)

st.sidebar.markdown("---")
st.sidebar.header("Prédiction personnalisée")
pred_year = st.sidebar.number_input("Année", value=datetime.now().year, step=1)
pred_month = st.sidebar.slider("Mois", min_value=1, max_value=12, value=mois)
pred_day = st.sidebar.slider("Jour", min_value=1, max_value=31, value=1)
pred_hour = st.sidebar.slider("Heure (0-23)", min_value=0, max_value=23, value=heure)
pred_weekday = st.sidebar.selectbox("Weekday (0=Lundi)", unique_weekdays, index=unique_weekdays.index(jour) if jour in unique_weekdays else 0)

# Main layout columns
col_map, col_right = st.columns([2, 1])

# Map
with col_map:
    st.subheader("Carte des compteurs — Montpellier")
    map_df = df.copy()
    # Option: filtrer par station sélectionnée
    if station_sel != "Tous":
        map_df = map_df[map_df["id"] == station_sel]

    # calculer moyenne valeur par station
    station_agg = map_df.groupby(["id", "lat", "lon"], dropna=True)["value"].mean().reset_index()
    station_agg = station_agg.dropna(subset=["lat", "lon"])
    if station_agg.empty:
        st.info("Aucune coordonnée valide trouvée pour les compteurs.")
    else:
        # pydeck
        view = pdk.ViewState(latitude=43.6, longitude=3.87, zoom=12, pitch=30)
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=station_agg,
            get_position=["lon", "lat"],
            get_radius="value * 10 + 100", 
            radius_scale=1,
            get_fill_color="[255 - value, value*2, 160, 180]",
            pickable=True,
            auto_highlight=True,
        )
        tooltip = {"html": "<b>id:</b> {id} <br/> <b>mean:</b> {value}", "style": {"backgroundColor": "white", "color": "black"}}
        r = pdk.Deck(layers=[scatter], initial_view_state=view, tooltip=tooltip)
        st.pydeck_chart(r)

with col_right:
    st.subheader("Prédiction & métriques")
    # Baseline
    baseline = df["value"].mean()
    if metrics:
        st.metric("RMSE (train)", f"{metrics.get('RMSE', 'n/a'):.2f}")
        st.metric("MAE (train)", f"{metrics.get('MAE', 'n/a'):.2f}")
    else:
        st.info("Métriques non disponibles (fichier metrics.json absent).")

    # Predict button
    X_input = build_input_df(pred_year, pred_month, pred_day, pred_hour, pred_weekday)
    st.write("Input construit pour le modèle :")
    st.dataframe(X_input)

    if model is None:
        st.warning("Aucun modèle chargé — impossible de prédire.")
    else:
        try:
            
            for c in X_input.columns:
                if X_input[c].dtype == object:
                    try:
                        X_input[c] = X_input[c].astype(int)
                    except Exception:
                        pass
            pred_val = model.predict(X_input)[0]
            st.metric("Prédiction (nombre de cyclistes)", f"{pred_val:.1f}")
            st.write(f"Baseline (moyenne historique) : {baseline:.2f}")

            # comparaison graphique rapide
            comp_df = pd.DataFrame({
                "Type": ["Prédiction", "Baseline"],
                "Valeur": [pred_val, baseline]
            })
            st.bar_chart(comp_df.set_index("Type"))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# Section d'exploration & histogrammes
st.markdown("---")
st.header("Exploration rapide des données")
# appliquer filtres
explore_df = df.copy()
if station_sel != "Tous":
    explore_df = explore_df[explore_df["id"] == station_sel]
explore_df = explore_df[(explore_df["weekday"] == jour) & (explore_df["month"] == mois) & (explore_df["hour"] == heure)]
st.write(f"Échantillon ({len(explore_df)} lignes) — filtres appliqués")
st.dataframe(explore_df.head(50))

st.markdown("### Distribution des valeurs (tous compteurs)")
st.bar_chart(df["value"].value_counts().sort_index())

# Footer
st.markdown("---")
st.write("Remarques :")
st.write("- Le CSV doit contenir une colonne `time` parsable en datetime.")
st.write("- `coordinates` doit contenir des couples lat,lon (ex : `(43.6, 3.87)` ).")
st.write("- Si le modèle attend d'autres colonnes ou un encodage particulier, adapte `MODEL_FEATURES_ORDER` et le `build_input_df`.")
