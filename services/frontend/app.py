import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration
st.set_page_config(
    page_title="🚴 Trafic Cyclable Montpellier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

DATA_CSV = "../training/pipeline/data/raw/df_brut.csv"
API_URL = "http://127.0.0.1:8000/predict"

#
@st.cache_data
def load_data(csv_path=DATA_CSV):
    try:
        df = pd.read_csv(csv_path, parse_dates=["time"])
    except ValueError:
        df = pd.read_csv(csv_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            raise FileNotFoundError("Le CSV doit contenir une colonne 'time'")
    
    for c in ["time", "value", "id", "coordinates"]:
        if c not in df.columns:
            raise KeyError(f"Colonne manquante : '{c}'")

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday

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
                if abs(lat) > 90:
                    lat, lon = lon, lat
                return (lat, lon)
            except Exception:
                return (np.nan, np.nan)
        return (np.nan, np.nan)

    coords = df["coordinates"].apply(parse_coord)
    df["lat"] = coords.apply(lambda t: t[0])
    df["lon"] = coords.apply(lambda t: t[1])

    return df

def call_api(year, month, day, hour, weekday):
    params = {
        "year": int(year),
        "month": int(month),
        "day": int(day),
        "hour": int(hour),
        "weekday": int(weekday)
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["prediction"][0]["forecast"] if data.get("prediction") else None
    except Exception as e:
        st.error(f" Erreur API : {e}")
        return None

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f" Erreur chargement données : {e}")
    st.stop()


st.markdown('<h1 class="main-header">🚴 Prévision du Trafic Cyclable — Montpellier</h1>', unsafe_allow_html=True)
st.markdown("**Explorez les données historiques et testez le modèle de prédiction en temps réel**")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([" Vue d'ensemble", " Prédictions", " Analyse temporelle", " Export"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(" Compteurs actifs", len(df["id"].unique()))
    with col2:
        st.metric(" Période couverte", f"{df['year'].min()} - {df['year'].max()}")
    with col3:
        st.metric(" Trafic moyen", f"{df['value'].mean():.1f}")
    with col4:
        st.metric(" Pic enregistré", f"{df['value'].max():.0f}")
    
    st.markdown("---")
    
    col_map, col_stats = st.columns([2, 1])
    
    with col_map:
        st.subheader(" Carte interactive des compteurs")
        
        station_agg = df.groupby(["id", "lat", "lon"], dropna=True).agg({
            "value": ["mean", "max", "count"]
        }).reset_index()
        station_agg.columns = ["id", "lat", "lon", "avg_value", "max_value", "count"]
        station_agg = station_agg.dropna(subset=["lat", "lon"])
        
        if station_agg.empty:
            st.info("Aucune coordonnée valide")
        else:
            view = pdk.ViewState(latitude=43.6, longitude=3.87, zoom=12, pitch=40)
            layer = pdk.Layer(
                "ColumnLayer",
                data=station_agg,
                get_position=["lon", "lat"],
                get_elevation="avg_value * 50",
                elevation_scale=1,
                radius=100,
                get_fill_color="[255 - avg_value, avg_value*2, 160, 200]",
                pickable=True,
                auto_highlight=True,
            )
            tooltip = {
                "html": "<b>🚴 Compteur {id}</b><br/>"
                        "Moyenne: {avg_value:.1f}<br/>"
                        "Max: {max_value:.0f}<br/>"
                        "Mesures: {count}",
                "style": {"backgroundColor": "white", "color": "black", "fontSize": "14px"}
            }
            r = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip)
            st.pydeck_chart(r, use_container_width=True)
    
    with col_stats:
        st.subheader(" Top 5 compteurs")
        top_stations = df.groupby("id")["value"].mean().sort_values(ascending=False).head(5)
        fig_top = px.bar(
            x=top_stations.values,
            y=top_stations.index,
            orientation='h',
            labels={'x': 'Trafic moyen', 'y': 'Compteur'},
            color=top_stations.values,
            color_continuous_scale='Viridis'
        )
        fig_top.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_top, use_container_width=True)
        
        st.subheader(" Distribution")
        fig_dist = px.histogram(df, x="value", nbins=50, labels={'value': 'Trafic'})
        fig_dist.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

with tab2:
    st.subheader(" Testez le modèle de prédiction")
    
    col_pred1, col_pred2 = st.columns([1, 1])
    
    with col_pred1:
        st.markdown("#### Configuration unique")
        pred_year = st.number_input("Année", value=datetime.now().year, step=1)
        pred_month = st.slider("Mois", 1, 12, datetime.now().month)
        pred_day = st.slider("Jour", 1, 31, datetime.now().day)
        pred_hour = st.slider("Heure", 0, 23, 12)
        
        weekday_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        pred_weekday = st.selectbox("Jour de la semaine", range(7), format_func=lambda x: weekday_names[x])
        
        if st.button(" Lancer la prédiction", type="primary"):
            with st.spinner("Calcul en cours..."):
                pred_val = call_api(pred_year, pred_month, pred_day, pred_hour, pred_weekday)
                
                if pred_val is not None:
                    st.session_state['last_prediction'] = pred_val
                    st.session_state['baseline'] = df['value'].mean()
    
    with col_pred2:
        st.markdown("#### Résultat")
        
        if 'last_prediction' in st.session_state:
            pred_val = st.session_state['last_prediction']
            baseline = st.session_state['baseline']
            diff = pred_val - baseline
            diff_pct = (diff / baseline) * 100
            
            st.markdown(f"""
            <div class="success-box">
                <h2 style='margin:0; color:#155724;'>🚴 {pred_val:.0f} cyclistes</h2>
                <p style='margin:0.5rem 0 0 0; color:#155724;'>
                    {'' if diff > 0 else ''} {diff:+.0f} vs baseline ({diff_pct:+.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparaison visuelle
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=['Baseline'], y=[baseline], name='Baseline', marker_color='lightgray'))
            fig_comp.add_trace(go.Bar(x=['Prédiction'], y=[pred_val], name='Prédiction', marker_color='#667eea'))
            fig_comp.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info(" Configurez les paramètres et lancez une prédiction")
    
    st.markdown("---")
    st.subheader(" Prédictions multiples (comparaison)")
    
    if st.checkbox("Activer le mode comparaison"):
        hours_to_compare = st.multiselect("Heures à comparer", range(24), default=[8, 12, 18])
        
        if st.button("Comparer ces heures"):
            predictions = []
            for h in hours_to_compare:
                pred = call_api(pred_year, pred_month, pred_day, h, pred_weekday)
                if pred is not None:
                    predictions.append({"Heure": f"{h}h", "Prédiction": pred})
            
            if predictions:
                df_comp = pd.DataFrame(predictions)
                fig = px.line(df_comp, x="Heure", y="Prédiction", markers=True)
                fig.update_traces(line_color='#667eea', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_comp, use_container_width=True)

with tab3:
    st.subheader(" Analyse des patterns temporels")
    
    selected_station = st.selectbox("Sélectionner un compteur", ["Tous"] + sorted(df["id"].unique().tolist()))
    
    df_filtered = df if selected_station == "Tous" else df[df["id"] == selected_station]
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("**Trafic moyen par heure**")
        hourly = df_filtered.groupby("hour")["value"].mean().reset_index()
        fig_hour = px.line(hourly, x="hour", y="value", markers=True, labels={'hour': 'Heure', 'value': 'Trafic moyen'})
        fig_hour.update_traces(line_color='#667eea', line_width=3)
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col_t2:
        st.markdown("**Trafic moyen par jour de semaine**")
        weekly = df_filtered.groupby("weekday")["value"].mean().reset_index()
        weekday_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        weekly["day_name"] = weekly["weekday"].apply(lambda x: weekday_names[x])
        fig_week = px.bar(weekly, x="day_name", y="value", color="value", color_continuous_scale='Viridis', labels={'day_name': 'Jour', 'value': 'Trafic moyen'})
        st.plotly_chart(fig_week, use_container_width=True)
    
    st.markdown("**Heatmap : Heure × Jour de la semaine**")
    heatmap_data = df_filtered.pivot_table(values="value", index="hour", columns="weekday", aggfunc="mean")
    fig_heat = px.imshow(heatmap_data, 
                         labels=dict(x="Jour", y="Heure", color="Trafic"),
                         x=[weekday_names[i] for i in range(7)],
                         color_continuous_scale='Viridis')
    st.plotly_chart(fig_heat, use_container_width=True)


with tab4:
    st.subheader(" Export des données")
    
    st.markdown("""
    <div class="info-box">
    Exportez les données filtrées ou les résultats de prédictions pour vos analyses externes.
    </div>
    """, unsafe_allow_html=True)
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("**Export données brutes**")
        export_station = st.selectbox("Compteur", ["Tous"] + sorted(df["id"].unique().tolist()), key="export")
        df_export = df if export_station == "Tous" else df[df["id"] == export_station]
        
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Télécharger CSV",
            data=csv_data,
            file_name=f"trafic_cyclable_{export_station}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col_e2:
        st.markdown("**Statistiques rapides**")
        st.write(f"Lignes : {len(df_export):,}")
        st.write(f"Moyenne : {df_export['value'].mean():.1f}")
        st.write(f"Médiane : {df_export['value'].median():.1f}")
        st.write(f"Écart-type : {df_export['value'].std():.1f}")

