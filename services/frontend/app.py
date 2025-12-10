import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client

# 1. Setup
dotenv_path = "/home/marwa/prevision-du-Trafic-Cyclable/.env"
load_dotenv(dotenv_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="🚴 Trafic Cyclable Montpellier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Styles CSS modernisés
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
   
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin: 1rem;
        background : #B0C4DE;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    
    .hero-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border: 2px solid #667eea;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #667eea;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        color: #333;
        box-shadow: 0 15px 40px rgba(17, 153, 142, 0.3);
        border: 3px solid #11998e;
        animation: pulse 2s infinite;
        margin: 1rem 0;
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .prediction-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #11998e;
        font-weight: 600;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        border: 2px solid #f093fb;
    }
    
    .info-card h3 {
        color: #f5576c;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #666;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stSelectbox, .stNumberInput, .stSlider {
        margin: 0.5rem 0;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    h3, h2, .stMarkdown h3, .stMarkdown h2 {
        color: #2d3748 !important;
    }
    
    p, .stMarkdown p {
        color: #4a5568 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        border: none;
        color: #2d3748;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 3. Supabase
@st.cache_resource
def init_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error(" Configuration Supabase manquante")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=600)
def load_best_counters() -> pd.DataFrame:
    supabase = init_supabase()
    resp = supabase.table("best_counters").select("*").execute()
    best_df = pd.DataFrame(resp.data or [])
    if best_df.empty:
        st.warning(" Aucun compteur dans la table best_counters")
    return best_df

@st.cache_data(ttl=600)
def load_forecast_data() -> pd.DataFrame:
    supabase = init_supabase()
    resp = supabase.table("forecast_data").select("*").execute()
    df = pd.DataFrame(resp.data or [])

    if df.empty:
        st.warning("Aucune donnée de prévision en base")
        return df

    # Convertir la date si nécessaire
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

@st.cache_data(ttl=600)
def load_data_from_supabase(table_name="historical_data") -> pd.DataFrame:
    supabase = init_supabase()
    resp = supabase.table(table_name).select("*").execute()
    df = pd.DataFrame(resp.data or [])
    if df.empty:
        return df

    if 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['time'] = pd.NaT

    if 'intensity' in df.columns and 'value' not in df.columns:
        df['value'] = df['intensity']

    for col in ['year','month','day','hour','weekday']:
        if col not in df.columns or df[col].isnull().all():
            if col != 'hour':
                df[col] = getattr(df['time'].dt, col if col != 'weekday' else 'weekday')
            else:
                df[col] = 0

    if 'coordinates' in df.columns and ('lat' not in df.columns or 'lon' not in df.columns):
        def parse_coord(x):
            if x is None: 
                return (np.nan, np.nan)
            if isinstance(x, (list, tuple)) and len(x) == 2:
                lat, lon = x
                if abs(lat) > 90: lat, lon = lon, lat
                return float(lat), float(lon)
            try:
                s = str(x).replace("(", "").replace(")", "").replace("[","").replace("]","").strip()
                parts = [float(p) for p in s.split(",")]
                if len(parts) != 2:
                    return (np.nan, np.nan)
                lat, lon = parts
                if abs(lat) > 90: lat, lon = lon, lat
                return lat, lon
            except:
                return (np.nan, np.nan)

        coords = df['coordinates'].apply(parse_coord)
        df['lat'] = coords.apply(lambda t: t[0])
        df['lon'] = coords.apply(lambda t: t[1])

    return df

# 4. API helper
def call_api_single(year, month, day, hour, weekday, counter_id=None, retries=1):
    params = {"year": year, "month": month, "day": day, "hour": hour, "weekday": weekday}
    if counter_id is not None:
        params["counter_id"] = str(counter_id)
    try:
        resp = requests.get(API_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("prediction", [{}])[0].get("forecast")
    except Exception as e:
        if retries > 0:
            return call_api_single(year, month, day, hour, weekday, counter_id, retries-1)
        st.error(f" Erreur API : {e}")
        return None

# 5. Chargement des données
with st.spinner(" Chargement des données..."):
    df = load_data_from_supabase()
if df.empty:
    st.error(" Aucune donnée disponible")
    st.stop()

# 6. Hero Header
st.markdown('<h1 class="hero-header">🚴 Trafic Cyclable Montpellier</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Analyse intelligente et prédictions en temps réel</p>', unsafe_allow_html=True)

# 7. Tabs
tab1, tab2, tab3, tab4 = st.tabs([" Dashboard", "Prédictions", "📈 Analyses", "💾 Export"])

# TAB 1: Dashboard
# TAB 1: Dashboard
with tab1:
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Compteurs Actifs</div>
            <div class="metric-value">{df['id'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Trafic Moyen</div>
            <div class="metric-value">{df['value'].mean():.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pic Maximum</div>
            <div class="metric-value">{df['value'].max():.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Période</div>
            <div class="metric-value">{df['year'].min()}-{df['year'].max()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header"> Statistiques Clés</h2>', unsafe_allow_html=True)

    

    
    # Carte
    station_agg = df.groupby(['id', 'lat', 'lon']).agg({'value':'mean'}).reset_index()
    station_agg.rename(columns={"value":"avg_value"}, inplace=True)
    station_agg.dropna(subset=['lat','lon'], inplace=True)
    
    if not station_agg.empty:
        fig_map = px.scatter_mapbox(
            station_agg,
            lat="lat",
            lon="lon",
            hover_name="id",
            hover_data={"avg_value": ":.0f"},
            size="avg_value",
            size_max=25,
            color="avg_value",
            color_continuous_scale="Turbo",
            zoom=12,
            height=550
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0,r=0,t=0,b=0),
            mapbox=dict(center=dict(lat=43.6119, lon=3.8772), zoom=12),
            dragmode="zoom",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Graphiques en deux colonnes
    st.markdown('<h2 class="section-header"> Statistiques Clés</h2>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    # --- Top 5 et distribution en deux colonnes ---
    col_left, col_right = st.columns(2)

    # Top 10 des compteurs
with col_left:
    top10 = df.groupby('id')['value'].mean().sort_values(ascending=False).head(10)
    if not top10.empty:
        fig_top = px.bar(
            top10.reset_index(),
            x='id',
            y='value',
            text='value',
            color='value',
            color_continuous_scale='Viridis',
            title="🏆 Top 10 des Compteurs par Trafic Moyen"
        )
        fig_top.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_top.update_layout(
            xaxis_title="Compteur ID",
            yaxis_title="Trafic Moyen",
            paper_bgcolor='#DCDCDC',
            plot_bgcolor='#DCDCDC',
            font=dict(color='black', size=12),
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=20, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.warning(" Aucun compteur disponible pour le Top 10")

    # Distribution
    with col_right:
        fig_dist = go.Figure(go.Histogram(
            x=df['value'],
            nbinsx=50,
            marker=dict(
                color='rgb(102, 126, 234)',
                line=dict(color='white', width=1)
            )
        ))
        fig_dist.update_layout(
            title=" Distribution du Trafic",
            xaxis_title="Trafic",
            yaxis_title="Nombre de mesures",
            height=400,
            showlegend=False,
            paper_bgcolor='#DCDCDC',
            plot_bgcolor='#DCDCDC',
            font=dict(color='black', size=12),
            margin=dict(l=20, r=20, t=40, b=40)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    

# TAB 2: Prédictions
# TAB 2: Prédictions
with tab2:
    st.markdown('<h2 class="section-header">📊 Prédictions enregistrées</h2>', unsafe_allow_html=True)

    forecast_df = load_forecast_data()
    if forecast_df.empty:
        st.error("La table forecast_data est vide.")
        st.stop()

    # Sélection compteur
    pred_counter_id = st.selectbox("📍 Choisir un compteur", sorted(forecast_df["counter_id"].unique()))

    # Filtrer
    df_sel = forecast_df[forecast_df["counter_id"] == pred_counter_id].sort_values("date")

    # Sélection date
    pred_date = st.date_input("📅 Choisir une date", df_sel["date"].min())

    # Récupérer la ligne
    row = df_sel[df_sel["date"] == pd.to_datetime(pred_date)]

    if row.empty or row["forecast"].iloc[0] is None:
        st.warning("Aucune prévision enregistrée pour ce jour.")
    else:
        pred_val = row["forecast"].iloc[0]
        baseline = df["value"].mean()

        # Vérification que pred_val est bien un nombre
        if pred_val is not None:
            diff = pred_val - baseline
            diff_pct = (diff / baseline * 100) if baseline != 0 else 0

            # Carte prédiction
            st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">Prévision enregistrée</div>
                <div class="prediction-value">{pred_val:.0f} 🚴</div>
                <div class="prediction-subtitle">
                    {diff:+.0f} cyclistes vs baseline ({diff_pct:+.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Graphique comparaison Baseline / Prédiction
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=['Baseline', 'Prédiction'],
                y=[baseline, pred_val],
                marker=dict(color=['lightgray', 'rgb(102, 126, 234)']),
                text=[f'{baseline:.0f}', f'{pred_val:.0f}'],
                textposition='auto'
            ))
            fig_comp.update_layout(
                height=300,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Tableau avec la prévision
            st.dataframe(row, use_container_width=True, height=200)
        else:
            st.warning("La valeur de prévision est invalide.")



# TAB 3: Analyses
with tab3:
    st.markdown('<h2 class="section-header">📈 Analyse Temporelle Avancée</h2>', unsafe_allow_html=True)
    
    selected_station = st.selectbox("📍 Sélectionner un compteur", ["🌐 Tous"] + sorted(df["id"].unique()))
    df_filtered = df if selected_station == "🌐 Tous" else df[df["id"]==selected_station]
    
    col1, col2 = st.columns(2)
    
    with col1:
        hourly = df_filtered.groupby("hour")["value"].mean().reset_index()
        fig_hour = go.Figure(go.Scatter(
            x=hourly["hour"],
            y=hourly["value"],
            mode='lines+markers',
            line=dict(color='rgb(102, 126, 234)', width=3),
            marker=dict(size=8, color='rgb(118, 75, 162)'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig_hour.update_layout(
            title="🕐 Trafic par Heure",
            xaxis_title="Heure",
            yaxis_title="Trafic Moyen",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col2:
        weekly = df_filtered.groupby("weekday")["value"].mean().reset_index()
        weekday_names_short = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        weekly["day_name"] = weekly["weekday"].apply(lambda x: weekday_names_short[x])
        fig_week = go.Figure(go.Bar(
            x=weekly["day_name"],
            y=weekly["value"],
            marker=dict(
                color=weekly["value"],
                colorscale='Viridis',
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.0f}' for v in weekly["value"]],
            textposition='auto'
        ))
        fig_week.update_layout(
            title="📅 Trafic par Jour",
            xaxis_title="Jour",
            yaxis_title="Trafic Moyen",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_week, use_container_width=True)
    
    # Heatmap
    st.markdown('<h2 class="section-header"> Intensité du Trafic  </h2>', unsafe_allow_html=True)
    heatmap_data = df_filtered.pivot_table(values="value", index="hour", columns="weekday", aggfunc="mean").fillna(0)
    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=weekday_names_short,
        y=list(range(24)),
        colorscale='Viridis',
        colorbar=dict(title="Trafic")
    ))
    fig_heat.update_layout(
        title="Intensité du Trafic (Heure × Jour)",
        xaxis_title="Jour de la Semaine",
        yaxis_title="Heure de la Journée",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    

# TAB 4: Export
with tab4:
    st.markdown('<h2 class="section-header">💾 Export de Données</h2>', unsafe_allow_html=True)
    
    export_station = st.selectbox("📍 Sélectionner un compteur", ["🌐 Tous"] + sorted(df["id"].unique()), key="export")
    df_export = df if export_station == "🌐 Tous" else df[df["id"]==export_station]
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Lignes
    col1.markdown(f"""
    <div style="
        background:white; 
        padding:1rem; 
        border-radius:15px; 
        text-align:center; 
        box-shadow:0 10px 30px rgba(102, 126, 234, 0.3);
        border:2px solid #667eea;">
        <div style="color:#667eea; font-weight:600;"> Lignes</div>
        <div style="color:#764ba2; font-size:1.8rem; font-weight:700;">{len(df_export):,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Moyenne
    col2.markdown(f"""
    <div style="
        background:white; 
        padding:1rem; 
        border-radius:15px; 
        text-align:center; 
        box-shadow:0 10px 30px rgba(102, 126, 234, 0.3);
        border:2px solid #667eea;">
        <div style="color:#667eea; font-weight:600;">📈 Moyenne</div>
        <div style="color:#38ef7d; font-size:1.8rem; font-weight:700;">{df_export['value'].mean():.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Médiane
    
    
    col3.markdown(f"""
    <div style="
        background:white; 
        padding:1rem; 
        border-radius:15px; 
        text-align:center; 
        box-shadow:0 10px 30px rgba(102, 126, 234, 0.3);
        border:2px solid #667eea;">
        <div style="color:#667eea; font-weight:600;">📌 Médiane</div>
        <div style="color:#f093fb; font-size:1.8rem; font-weight:700;">{df_export['value'].median():.1f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Valeur maximale
    col4.markdown(f"""
    <div style="
        background:white; 
        padding:1rem; 
        border-radius:15px; 
        text-align:center; 
        box-shadow:0 10px 30px rgba(102, 126, 234, 0.3);
        border:2px solid #667eea;">
        <div style="color:#667eea; font-weight:600;"> Maximum</div>
        <div style="color:#f5576c; font-size:1.8rem; font-weight:700;">{df_export['value'].max():.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "📥 Télécharger en CSV",
        df_export.to_csv(index=False).encode("utf-8"),
        file_name=f"export_compteur_{export_station}.csv",
        mime="text/csv",
        use_container_width=True
        
    )
    
   
    
    st.markdown("### 📋 Aperçu des Données")
    st.dataframe(df_export.head(100), use_container_width=True, height=400)