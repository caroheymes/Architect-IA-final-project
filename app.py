import os
import time

import numpy as np
import pandas as pd
import streamlit as st


def log_time(tag):
    print(
        f"[ANTIGRAVITY_LOG] [{time.strftime('%Y-%m-%d %H:%M:%S')}.{int((time.time() % 1) * 1000):03d}] {tag}",
        flush=True,
    )


log_time("Script import/startup start")
log_time("Imports completed")

st.set_page_config(page_title="LyonFlow - Traffic Prediction", page_icon="🚦", layout="wide")

log_time("Page config set")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
    
    /* Apply Outfit font to the entire application */
    .stApp, .main-title, .subtitle, .metric-card {
        font-family: 'Outfit', -apple-system, sans-serif !important;
    }
    
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
        border-color: #3B82F6;
    }
    
    .metric-card h3 {
        font-size: 1.1rem;
        color: #4B5563;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-card h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        font-size: 0.9rem;
        color: #6B7280;
        margin: 0;
    }

    /* --- LyonFlow Square Map Layout System --- */
    
    /* Target the parent vertical block that contains our anchor */
    div[data-testid="stVerticalBlock"]:has(#map-section) {
        width: 100% !important;
        margin: 0 auto !important;
    }
    
    /* Target all map widget containers immediately following the anchor */
    div[data-testid="stVerticalBlock"]:has(#map-section) div.element-container:has(iframe),
    div[data-testid="stVerticalBlock"]:has(#map-section) div.element-container:has(canvas),
    div[data-testid="stVerticalBlock"]:has(#map-section) div.element-container:has(.stPydeckChart),
    div[data-testid="stVerticalBlock"]:has(#map-section) div.element-container:has([data-testid="stDeckGlChart"]),
    div[data-testid="stVerticalBlock"]:has(#map-section) div.element-container:has([data-testid="stPlotlyChart"]),
    div[data-testid="stVerticalBlock"]:has(#map-section) div.element-container:has([data-testid="stMap"]),
    div[data-testid="stVerticalBlock"]:has(#map-section) div.element-container:has(.stMap),
    div[data-testid="stVerticalBlock"]:has(#map-section) div[data-testid="stElementContainer"]:has(iframe),
    div[data-testid="stVerticalBlock"]:has(#map-section) div[data-testid="stElementContainer"]:has(canvas),
    div[data-testid="stVerticalBlock"]:has(#map-section) div[data-testid="stElementContainer"]:has(.stPydeckChart),
    div[data-testid="stVerticalBlock"]:has(#map-section) div[data-testid="stElementContainer"]:has([data-testid="stDeckGlChart"]),
    div[data-testid="stVerticalBlock"]:has(#map-section) div[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]),
    div[data-testid="stVerticalBlock"]:has(#map-section) div[data-testid="stElementContainer"]:has([data-testid="stMap"]),
    div[data-testid="stVerticalBlock"]:has(#map-section) div[data-testid="stElementContainer"]:has(.stMap) {
        aspect-ratio: 1 / 1 !important;
        width: 100% !important;
        height: auto !important;
        min-height: unset !important;
        max-height: unset !important;
    }

    /* Force all inner canvas, iframe, and chart container elements to be square */
    div[data-testid="stVerticalBlock"]:has(#map-section) canvas,
    div[data-testid="stVerticalBlock"]:has(#map-section) .mapboxgl-canvas,
    div[data-testid="stVerticalBlock"]:has(#map-section) .js-plotly-plot,
    div[data-testid="stVerticalBlock"]:has(#map-section) .plotly,
    div[data-testid="stVerticalBlock"]:has(#map-section) div.deckgl-container,
    div[data-testid="stVerticalBlock"]:has(#map-section) div.stDeckGlChart,
    div[data-testid="stVerticalBlock"]:has(#map-section) iframe {
        aspect-ratio: 1 / 1 !important;
        width: 100% !important;
        height: 100% !important;
        min-height: unset !important;
        max-height: unset !important;
    }

    /* Global fallback square aspect-ratio rules for older browsers (no :has required) */
    div[data-testid="stDeckGlChart"],
    div.stPydeckChart,
    div[data-testid="stPlotlyChart"],
    div[data-testid="stMap"],
    div.stMap,
    div.deckgl-container,
    .js-plotly-plot,
    .plotly,
    canvas.mapboxgl-canvas,
    div.stPydeckChart canvas,
    div[data-testid="stDeckGlChart"] canvas {
        aspect-ratio: 1 / 1 !important;
        width: 100% !important;
    }

    /* Style and refine Streamlit Tabs to look like premium compact card tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #F1F5F9 !important; /* Premium Slate-100 */
        padding: 6px !important;
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        margin-bottom: 25px !important;
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: space-evenly !important; /* Distribute tabs evenly */
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.04) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem !important; /* Elegantly proportioned tab headers matching body/metric card scales */
        font-weight: 600 !important;
        padding-left: 20px !important; /* Elegant compact padding */
        padding-right: 20px !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        border-radius: 8px !important; /* Smooth matching proportions */
        border: none !important;
        background-color: transparent !important;
        color: #475569 !important; /* Premium Slate-600 */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        height: auto !important;
        flex-grow: 1 !important; /* Make tabs distribute width nicely */
        text-align: center !important;
        
        /* Shorter tabs don't require wrapping limits */
        white-space: nowrap !important; /* Shorter titles fit beautifully on one line */
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #FFFFFF !important;
        color: #1E3A8A !important; /* Blue-900 */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
        transform: translateY(-1px) !important;
    }
    
    .stTabs [aria-selected="true"] {
        font-size: 1.1rem !important; /* Premium active tab header */
        font-weight: 700 !important;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important; /* Royal/Lyon Blue gradient */
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
        transform: translateY(-0.5px) !important;
    }
    
    /* Remove default bottom line of Streamlit tabs */
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Force inner texts of tab buttons to inherit state-driven styles (color, size, weight) */
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab"] label {
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
    }
    
    /* Explicit hover override for children of inactive tab buttons */
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span,
    .stTabs [data-baseweb="tab"]:hover div,
    .stTabs [data-baseweb="tab"]:hover label {
        color: #1E3A8A !important; /* Force beautiful deep blue on hover of inactive tabs */
    }
    
    /* Explicit active selected hover states for children (must remain brilliant white!) */
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div,
    .stTabs [aria-selected="true"] label,
    .stTabs [aria-selected="true"]:hover p,
    .stTabs [aria-selected="true"]:hover span,
    .stTabs [aria-selected="true"]:hover div,
    .stTabs [aria-selected="true"]:hover label {
        color: #FFFFFF !important; /* Force brilliant white on selected tab hover */
    }
    
    /* --- Styles for Tab Content Body Text --- */
    /* Target all paragraphs and text inside tabs to be beautifully readable, clean, and professional */
    .stTabs [role="tabpanel"] p, 
    .stTabs [role="tabpanel"] [data-testid="stMarkdownContainer"] p,
    .stTabs [role="tabpanel"] [data-testid="stMarkdownContainer"] span {
        font-size: 1.0rem !important; /* Compact, readable body text size! */
        line-height: 1.5 !important; /* Comfortable spacing */
        color: #334155 !important; /* Warm Slate-700 for high-end readability */
    }
    
    /* Target lists inside tabs */
    .stTabs [role="tabpanel"] li,
    .stTabs [role="tabpanel"] [data-testid="stMarkdownContainer"] li {
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        color: #334155 !important;
        margin-bottom: 6px !important;
    }
    
    /* Target subheaders inside tabs (e.g. st.subheader) */
    .stTabs [role="tabpanel"] h3,
    .stTabs [role="tabpanel"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.45rem !important; /* Balanced subheadings */
        font-weight: 700 !important;
        color: #0F172A !important; /* Deep Slate-900 */
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }
    
    /* Target secondary headers inside tabs (like st.markdown("### ...")) */
    .stTabs [role="tabpanel"] h2,
    .stTabs [role="tabpanel"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin-top: 25px !important;
        margin-bottom: 12px !important;
    }

    /* Target inline code blocks inside tabs */
    .stTabs [role="tabpanel"] code,
    .stTabs [role="tabpanel"] [data-testid="stMarkdownContainer"] code {
        font-size: 0.95rem !important;
        background-color: #F1F5F9 !important;
        padding: 3px 6px !important;
        border-radius: 6px !important;
        font-family: 'Fira Code', 'Courier New', monospace !important;
    }
    
    /* Target Streamlit standard alert/notification boxes inside tabs */
    .stTabs [role="tabpanel"] [data-testid="stNotification"] p,
    .stTabs [role="tabpanel"] [data-testid="stNotification"] div,
    .stTabs [role="tabpanel"] [data-testid="stAlert"] p {
        font-size: 1.0rem !important;
        line-height: 1.4 !important;
        font-weight: 500 !important;
    }
    
    /* Target LaTeX formulas inside tabs */
    .stTabs [role="tabpanel"] .katex {
        font-size: 1.0rem !important;
    }

    /* --- Overrides for Specific Widgets to prevent bloat --- */
    
    /* Restore standard compact sizes for Streamlit st.metric widgets inside tab panels */
    .stTabs [role="tabpanel"] [data-testid="stMetricValue"],
    .stTabs [role="tabpanel"] [data-testid="stMetricValue"] div,
    .stTabs [role="tabpanel"] [data-testid="stMetricValue"] p,
    .stTabs [role="tabpanel"] [data-testid="stMetricValue"] span {
        font-size: 1.6rem !important; /* Keep actual metric value prominent and compact */
        font-weight: 700 !important;
        color: #1E293B !important;
    }
    
    .stTabs [role="tabpanel"] [data-testid="stMetricLabel"],
    .stTabs [role="tabpanel"] [data-testid="stMetricLabel"] div,
    .stTabs [role="tabpanel"] [data-testid="stMetricLabel"] p,
    .stTabs [role="tabpanel"] [data-testid="stMetricLabel"] span {
        font-size: 0.85rem !important; /* Neat, compact label size */
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    /* Target Streamlit standard captions specifically inside tabs to keep them small and elegant */
    .stTabs [role="tabpanel"] [data-testid="stCaptionContainer"],
    .stTabs [role="tabpanel"] [data-testid="stCaptionContainer"] p,
    .stTabs [role="tabpanel"] [data-testid="stCaptionContainer"] div,
    .stTabs [role="tabpanel"] [data-testid="stCaptionContainer"] span {
        font-size: 0.825rem !important; /* Elegant compact caption size */
        color: #64748B !important; /* Muted Slate-500 */
        line-height: 1.4 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🚦 LyonFlow</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Plateforme temps réel de prédiction de trafic - Métropole de Lyon</div>',
    unsafe_allow_html=True,
)

# Database Connection Status
DATABASE_URL = None

st.sidebar.header("Configuration & statut")
st.sidebar.success("🟢 inférence sur csv. Stockage Postgresql (architecture médaillon).")
st.sidebar.caption("Les données sont chargées localement en moins de 0.05s.")

st.sidebar.markdown("---")
st.sidebar.info("""
    **LyonFlow Stack**:
    - **Orchestration**: Apache Airflow
    - **Calcul Distribué**: Ray Core
    - **Stockage**: Postgresql
""")


def load_street_names(db_url=None):
    # Try local CSV first
    csv_path = "data/out/street_names.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                df["properties_twgid"] = df["properties_twgid"].astype(int)
                return df[["properties_twgid", "properties_libelle"]]
        except Exception:
            pass
    return pd.DataFrame(columns=["properties_twgid", "properties_libelle"])


def load_predictions_data(db_url=None):
    # Try local CSV first
    csv_path = "data/out/predictions_traffic.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df, "Fichiers CSV (Dag Airflow)"
        except Exception:
            pass
    return None, None


def parse_linestring_coords(wkt_str):
    try:
        if not wkt_str or not isinstance(wkt_str, str):
            return None
        content = wkt_str.replace("LINESTRING", "").replace("(", "").replace(")", "").strip()
        coords = [c.strip().split() for c in content.split(",")]
        return [[float(c[0]), float(c[1])] for c in coords if len(c) >= 2]
    except Exception:
        return None


def parse_wkt_centroid(wkt_str):
    try:
        if not wkt_str or not isinstance(wkt_str, str):
            return None, None
        content = wkt_str.replace("LINESTRING", "").replace("(", "").replace(")", "").strip()
        coords = [c.strip().split() for c in content.split(",")]
        lats = [float(c[1]) for c in coords if len(c) >= 2]
        lons = [float(c[0]) for c in coords if len(c) >= 2]
        if lats and lons:
            return sum(lats) / len(lats), sum(lons) / len(lons)
    except Exception:
        pass
    return None, None


def get_processed_predictions_cached(db_url=None):
    log_time("get_processed_predictions_cached start")
    df_preds, data_source = load_predictions_data(db_url)
    log_time(f"load_predictions_data complete (source={data_source})")
    if df_preds is None or df_preds.empty:
        log_time("load_predictions_data empty or None")
        return None, None, None

    # Standardize types and add calculated columns
    df_preds["prediction_timestamp"] = pd.to_datetime(df_preds["prediction_timestamp"])
    df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"])
    df_preds["speed_diff"] = df_preds["predicted_speed"] - df_preds["real_speed"]

    # Keep ONLY the latest prediction run/cycle to prevent duplicate lines and show accurate current status
    latest_run = df_preds["prediction_timestamp"].max()
    df_preds = df_preds[df_preds["prediction_timestamp"] == latest_run].copy()

    # Format the latest prediction timestamp to Lyon local time (Europe/Paris) for display
    try:
        latest_run_dt = pd.to_datetime(latest_run)
        if latest_run_dt.tzinfo is None:
            latest_run_local = latest_run_dt.tz_localize("UTC").tz_convert("Europe/Paris")
        else:
            latest_run_local = latest_run_dt.tz_convert("Europe/Paris")
        latest_run_display = latest_run_local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        latest_run_display = str(latest_run)

    # Calculate centroids
    log_time("Calculating centroids from geometry_wgs84_wkt...")
    centroids = df_preds["geometry_wgs84_wkt"].apply(parse_wkt_centroid)
    df_preds["latitude"] = [c[0] for c in centroids]
    df_preds["longitude"] = [c[1] for c in centroids]
    log_time("Centroids calculated")

    # Drop rows without coordinates
    df_preds = df_preds.dropna(subset=["latitude", "longitude"])

    # Map twgid to human-readable street names
    log_time("Loading street names...")
    df_streets = load_street_names(db_url)
    df_preds["properties_twgid"] = df_preds["properties_twgid"].astype(int)
    log_time("Street names loaded, merging...")

    # Merge to join street names from the loaded DataFrame
    df_preds = df_preds.merge(df_streets, on="properties_twgid", how="left")
    df_preds["nom_rue"] = df_preds["properties_libelle"].fillna(
        df_preds["properties_twgid"].apply(lambda x: f"Segment {x}")
    )
    log_time("Merge complete")

    # Pre-parse LINESTRING path coordinates
    log_time("Pre-parsing LINESTRING coords...")
    df_preds["parsed_path_coords"] = df_preds["geometry_wgs84_wkt"].apply(parse_linestring_coords)
    log_time("Pre-parsing complete")

    return df_preds, data_source, latest_run_display


def load_evidently_report(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def plot_training_curves(df_metrics):
    """Génère la figure Plotly des courbes d'apprentissage (perte + MAE).

    Crée une figure à deux sous-graphes côte à côte :
      - à gauche : la perte MSE standardisée d'entraînement par époque
      - à droite : la MAE de validation en km/h par époque

    Args:
        df_metrics (pandas.DataFrame): Doit contenir les colonnes
        `Epoch`, `Train Loss (std)` et `Test MAE (km/h)`.

    Returns:
        plotly.graph_objects.Figure: Figure Plotly prête à être passée à
        `st.plotly_chart`.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("📉 Perte d'entraînement (Perte Normalisée / MSE std)", "📈 MAE de validation (km/h)"),
    )

    # 1. Train Loss Curve
    fig.add_trace(
        go.Scatter(
            x=df_metrics["Epoch"],
            y=df_metrics["Train Loss (std)"],
            mode="lines+markers",
            name="Perte d'entraînement (std)",
            line=dict(color="#1E3A8A", width=3),
            marker=dict(size=6, color="#1E3A8A"),
            hovertemplate="Époque %{x}<br>Perte std: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Test MAE Curve
    fig.add_trace(
        go.Scatter(
            x=df_metrics["Epoch"],
            y=df_metrics["Test MAE (km/h)"],
            mode="lines+markers",
            name="MAE de validation (km/h)",
            line=dict(color="#10B981", width=3),
            marker=dict(size=6, color="#10B981"),
            hovertemplate="Époque %{x}<br>MAE: %{y:.2f} km/h<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=50, b=30),
    )

    fig.update_xaxes(title_text="Époque", gridcolor="#E5E7EB", row=1, col=1)
    fig.update_xaxes(title_text="Époque", gridcolor="#E5E7EB", row=1, col=2)
    fig.update_yaxes(title_text="Perte standardisée", gridcolor="#E5E7EB", row=1, col=1)
    fig.update_yaxes(title_text="MAE (km/h)", gridcolor="#E5E7EB", row=1, col=2)
    return fig


def load_training_history_from_mlflow(run_id):
    """Récupère l'historique des métriques depuis le serveur MLflow pour un run donné.

    Args:
        run_id (str): ID du run MLflow.

    Returns:
        pandas.DataFrame | None: DataFrame contenant Epoch, Train Loss (std) et Test MAE (km/h),
        ou None si la récupération échoue.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = MlflowClient()

        # Récupérer l'historique de train_loss_std et test_mae_kmh
        train_loss_history = client.get_metric_history(run_id, "train_loss_std")
        test_mae_history = client.get_metric_history(run_id, "test_mae_kmh")

        if not train_loss_history or not test_mae_history:
            return None

        epochs = [m.step for m in train_loss_history]
        train_losses = [m.value for m in train_loss_history]
        test_maes = [m.value for m in test_mae_history]

        df = pd.DataFrame({
            "Epoch": epochs,
            "Train Loss (std)": train_losses,
            "Test MAE (km/h)": test_maes
        })
        # Trier par époque pour être sûr de l'ordre
        df = df.sort_values("Epoch").reset_index(drop=True)
        return df
    except Exception as e:
        log_time(f"⚠️ Impossible de récupérer les métriques depuis MLflow ({e})")
        return None


# Load predictions data first (from cached local CSV, fast and non-blocking)
log_time("Before get_processed_predictions_cached")
df_preds, data_source, latest_run_display = get_processed_predictions_cached(DATABASE_URL)
log_time("After get_processed_predictions_cached")

# Extract default model and run info from predictions dataframe columns if they exist
df_model_name = "STGCN_V2_AdamW"
df_run_name = "STGCN_v2_20260603_002414"
df_run_id = "a368b69d77134047b461ea001a3cc6dd"

if df_preds is not None and not df_preds.empty:
    if "model_name" in df_preds.columns:
        df_model_name = str(df_preds["model_name"].iloc[0])
    if "run_name" in df_preds.columns:
        df_run_name = str(df_preds["run_name"].iloc[0])
    if "run_id" in df_preds.columns:
        df_run_id = str(df_preds["run_id"].iloc[0])

# Active model metrics (from local CSV columns or fallback defaults, 100% offline)
selected_run_id = df_run_id
selected_run_name = df_run_name
selected_run_status = "FINISHED"
selected_run_model = df_model_name

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Modèle de prédiction actif")
st.sidebar.markdown(f"""
- **Modèle** : `{selected_run_model}`
- **Entraînement** : `{selected_run_name}`
- **ID de Run** : `{selected_run_id[:8]}`
- **Statut** : 🟢 Terminé (CSV Local)
""")

# Ordered Tabs as requested by user
log_time("Setting up tabs...")
tab_pred, tab_obs, tab_err, tab_curves = st.tabs(["Prévisions", "Observabilité", "Erreurs", "Apprentissage"])

with tab_curves:
    log_time("Entering tab_curves block")
    st.subheader("📈 Courbes d'apprentissage (évolution de la perte & MAE par époque)")
    st.markdown("""
    Visualisez ci-dessous l'évolution de la **perte d'entraînement normalisée (MSE)** et de l'**erreur absolue moyenne (MAE)** de validation (exprimée en km/h) calculées à chaque époque.
    """)

    # Essayer de charger dynamiquement les métriques réelles depuis MLflow
    df_metrics_mlflow = load_training_history_from_mlflow(selected_run_id)

    if df_metrics_mlflow is not None and not df_metrics_mlflow.empty:
        st.success(f"🟢 Courbes d'apprentissage réelles chargées dynamiquement depuis MLflow (ID du Run : `{selected_run_id[:8]}`).")
        fig_curves = plot_training_curves(df_metrics_mlflow)
        st.plotly_chart(fig_curves, use_container_width=True)
    else:
        ideal_plot_path = "trash/ideal.png"
        fallback_plot_path = "trash/model_metrics.png"

        if os.path.exists(ideal_plot_path):
            st.info("💡 Chargement du graphique de performance historique de référence (Cible finale de production).")
            st.image(
                ideal_plot_path,
                caption="Courbe d'apprentissage de référence (Idéal de production)",
                use_container_width=True,
            )
        elif os.path.exists(fallback_plot_path):
            st.warning("⚠️ Chargement des métriques de référence initiales.")
            st.image(
                fallback_plot_path, caption="Courbe d'apprentissage initiale (STGCN de base)", use_container_width=True
            )
        else:
            st.error("❌ Aucun historique de performance n'a pu être trouvé sur le disque local ou sur MLflow.")

with tab_err:
    log_time("Entering tab_err block")
    st.subheader("Analyse d'erreur stratifiée (dernier modèle)")
    st.markdown("""
    Cette vue présente l'évaluation fine de la précision réelle du modèle découpée en 4 analyses clés pour garantir l'honnêteté et la transparence scientifique :
    1. **MAE par tranche de vitesse** : Évaluation selon l'état du trafic (embouteillages/ralentissements vs trafic fluide).
    2. **Biais de prédiction systématique** : Détection des zones de sur-estimation ou de sous-estimation de la vitesse.
    3. **Dispersion et Incertitude** : Analyse de l'écart-type des prédictions et des résidus d'erreurs.
    4. **Boîtes à moustaches (Boxplots)** : Distribution statistique des vitesses prédites par rapport à la réalité terrain.
    """)

    latest_plot_path = "data/stratified_error_analysis.png"
    if not os.path.exists(latest_plot_path):
        latest_plot_path = "data/out/stratified_error_analysis.png"
    if not os.path.exists(latest_plot_path):
        latest_plot_path = "models/stratified_error_analysis.png"

    if os.path.exists(latest_plot_path):
        st.success("🟢 Diagnostic stratifié d'erreur récupéré localement.")
        st.image(latest_plot_path, caption="Analyse d'erreur stratifiée - Génération locale", use_container_width=True)
    else:
        st.info("💡 Le diagnostic d'erreur stratifiée sera généré lors du prochain entraînement final de production.")

with tab_obs:
    log_time("Entering tab_obs block")
    st.subheader("📊 Observabilité du modèle & dérive temporelle (Evidently AI)")
    st.markdown("""
    Ce tableau de bord d'observabilité compare en continu la précision prédictive du modèle STGCN 
    sur la tranche horaire de la journée (08h00 à 20h00) entre la veille (Référence, $J-1$) 
    et aujourd'hui (Audit, $J$).
    """)

    report_html_path = "data/out/monitoring_report_morning.html"
    report_json_path = "data/out/monitoring_metrics_morning.json"

    # Lecture dynamique des métriques clés
    mae_val = None
    rmse_val = None
    mape_val = None
    r2_val = None
    drift_p_val = None
    drift_detected = None
    drift_threshold = 0.05

    if os.path.exists(report_json_path):
        try:
            import json

            with open(report_json_path, encoding="utf-8") as fj:
                metrics_data = json.load(fj)

            for metric in metrics_data.get("metrics", []):
                metric_type = metric.get("config", {}).get("type", "")
                metric_name = metric.get("metric_name", "")
                if "MAE" in metric_type:
                    mae_val = metric.get("value", {}).get("mean", None)
                elif "RMSE" in metric_type:
                    rmse_val = metric.get("value", None)
                elif "MAPE" in metric_type:
                    mape_val = metric.get("value", {}).get("mean", None)
                elif "R2Score" in metric_type:
                    r2_val = metric.get("value", None)
                elif "ValueDrift" in metric_type or "ValueDrift" in metric_name:
                    val_dict = metric.get("value", {})
                    if isinstance(val_dict, dict):
                        drift_p_val = val_dict.get("p_value") or val_dict.get("drift_score")
                        drift_detected = val_dict.get("drift_detected")
                        drift_threshold = val_dict.get("threshold", 0.05)
        except Exception as ej:
            st.sidebar.warning(f"⚠️ Impossible de parser les métriques JSON : {ej}")

    # Affichage du statut décisionnel de réentraînement (p-value)
    if drift_p_val is not None:
        st.markdown("### 🤖 Statut décisionnel de réentraînement (test Kolmogorov-Smirnov)")
        if drift_p_val < drift_threshold:
            # ALERTE DÉRIVE : Rouge vibrant premium
            st.markdown(
                f"""
                <div style="background-color: #FEE2E2; border-left: 6px solid #DC2626; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <h3 style="color: #991B1B; margin-top: 0; font-family: system-ui, -apple-system, sans-serif;">🚨 Alerte MLOps : dérive de données détectée !</h3>
                    <p style="color: #7F1D1D; font-size: 1.05rem; line-height: 1.6; font-family: system-ui, -apple-system, sans-serif;">
                        Le test statistique de <b>Kolmogorov-Smirnov</b> appliqué sur les vitesses réelles de la matinée (cible <i>target</i>) 
                        entre J-1 et J indique un décalage significatif des distributions.
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: 2rem; margin: 1.2rem 0; font-family: system-ui, -apple-system, sans-serif;">
                        <div>
                            <span style="font-size: 0.85rem; color: #991B1B; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">p-value calculée</span>
                            <strong style="font-size: 1.8rem; color: #DC2626;">{drift_p_val:.4e}</strong>
                        </div>
                        <div style="border-left: 1px solid #FFC5C5; padding-left: 2rem;">
                            <span style="font-size: 0.85rem; color: #991B1B; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">Seuil critique (α)</span>
                            <strong style="font-size: 1.8rem; color: #7F1D1D;">{drift_threshold:.2f}</strong>
                        </div>
                        <div style="border-left: 1px solid #FFC5C5; padding-left: 2rem;">
                            <span style="font-size: 0.85rem; color: #991B1B; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">Action système</span>
                            <strong style="font-size: 1.15rem; color: #B91C1C; background-color: #FEE2E2; padding: 0.4rem 0.8rem; border-radius: 0.375rem; border: 1px solid #DC2626; display: inline-block; margin-top: 0.3rem; font-weight: bold;">🔴 Réentraînement requis</strong>
                        </div>
                    </div>
                    <p style="color: #7F1D1D; margin-bottom: 0; font-style: italic; font-size: 0.95rem; font-family: system-ui, -apple-system, sans-serif;">
                        ⚠️ La dynamique du réseau routier a changé par rapport à la veille. Le modèle actuel risque de perdre en précision. Un réentraînement automatique via le DAG d'orchestration Airflow est préconisé.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            # STATUT STABLE : Vert vibrant premium
            st.markdown(
                f"""
                <div style="background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <h3 style="color: #065F46; margin-top: 0; font-family: system-ui, -apple-system, sans-serif;">✅ Statut MLOps : distributions stables</h3>
                    <p style="color: #064E3B; font-size: 1.05rem; line-height: 1.6; font-family: system-ui, -apple-system, sans-serif;">
                        Le test statistique de <b>Kolmogorov-Smirnov</b> appliqué sur les vitesses réelles de la matinée (cible <i>target</i>) 
                        ne révèle aucun décalage de distribution significatif entre J-1 et J.
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: 2rem; margin: 1.2rem 0; font-family: system-ui, -apple-system, sans-serif;">
                        <div>
                            <span style="font-size: 0.85rem; color: #065F46; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">p-value calculée</span>
                            <strong style="font-size: 1.8rem; color: #10B981;">{drift_p_val:.4f}</strong>
                        </div>
                        <div style="border-left: 1px solid #A7F3D0; padding-left: 2rem;">
                            <span style="font-size: 0.85rem; color: #065F46; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">Seuil critique (α)</span>
                            <strong style="font-size: 1.8rem; color: #064E3B;">{drift_threshold:.2f}</strong>
                        </div>
                        <div style="border-left: 1px solid #A7F3D0; padding-left: 2rem;">
                            <span style="font-size: 0.85rem; color: #065F46; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">Décision système</span>
                            <strong style="font-size: 1.15rem; color: #047857; background-color: #D1FAE5; padding: 0.4rem 0.8rem; border-radius: 0.375rem; border: 1px solid #10B981; display: inline-block; margin-top: 0.3rem; font-weight: bold;">🟢 Production active (OK)</strong>
                        </div>
                    </div>
                    <p style="color: #064E3B; margin-bottom: 0; font-style: italic; font-size: 0.95rem; font-family: system-ui, -apple-system, sans-serif;">
                        ℹ️ Le comportement général du trafic routier reste en adéquation avec les données d'apprentissage historiques. Aucun réentraînement n'est requis.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Affichage des cartes d'indicateurs clés de performance
    if mae_val is not None or rmse_val is not None or r2_val is not None:
        st.markdown("### 🏆 Indicateurs de performance de l'audit (matinée courante)")
        mc1, mc2, mc3, mc4 = st.columns(4)

        with mc1:
            if mae_val is not None:
                st.metric(
                    label="MAE (erreur absolue moyenne)",
                    value=f"{mae_val:.2f} km/h",
                    delta="Amélioration" if mae_val < 3 else None,
                    delta_color="normal",
                )
        with mc2:
            if rmse_val is not None:
                st.metric(label="RMSE (erreur quadratique moyenne)", value=f"{rmse_val:.2f} km/h")
        with mc3:
            if mape_val is not None:
                st.metric(label="MAPE (erreur moyenne en %)", value=f"{mape_val:.2f} %")
        with mc4:
            if r2_val is not None:
                st.metric(
                    label="R² score (coefficient de détermination)",
                    value=f"{r2_val:.4f}",
                    delta="Performance optimale" if r2_val > 0.8 else None,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    if os.path.exists(report_html_path):
        import streamlit.components.v1 as components

        st.success("🟢 Rapport d'observabilité de la pointe du matin (07h00 - 10h00) chargé avec succès.")

        show_report = st.checkbox("Afficher le rapport d'observabilité interactif complet (Evidently AI)", value=True)
        if show_report:
            html_content = load_evidently_report(report_html_path)
            components.html(html_content, height=1200, scrolling=True)
        else:
            st.info("💡 Cochez la case ci-dessus pour charger le rapport interactif complet (taille : 3.8 Mo).")
    else:
        st.info(
            "💡 Le rapport de monitoring Evidently est en cours de génération ou sera produit automatiquement par le run quotidien d'Airflow."
        )
        st.image(
            "https://raw.githubusercontent.com/evidentlyai/evidently/main/docs/book/_static/evidently_logo.png",
            width=300,
        )

with tab_pred:
    log_time("Entering tab_pred block")
    st.subheader("🗺️ Visualisation spatiotemporelle & prédictions de trafic")
    st.markdown("""
    Cette carte interactive affiche l'état actuel et prévu des segments routiers de la Métropole de Lyon.
    Vous pouvez filtrer par horizon temporel de prédiction et choisir la métrique à visualiser.
    """)

    if df_preds is not None and not df_preds.empty:
        # Display Data Source Status Badge
        st.caption(
            f"Source des données cartographiques : **{data_source}** • Dernière prédiction : `{latest_run_display}` (Heure de Lyon)"
        )

        # 2. Filters Layout
        f_col1, f_col2, f_col3 = st.columns([1, 1.5, 1.5])

        with f_col1:
            # Select horizon (default to 30 min)
            horizons_avail = sorted(df_preds["horizon_minutes"].unique())
            horizon_labels = {30: "🚀 +30 min", 60: "🚀 +1 heure (+60 min)", 180: "🚀 +3 heures (+180 min)"}
            horizon_sel = st.selectbox(
                "Horizon temporel :", horizons_avail, format_func=lambda x: horizon_labels.get(x, f"+{x} min")
            )

        with f_col2:
            # Select Metric to show
            metric_options = {
                "predicted_speed": "🚦 Vitesse prédite (km/h)",
                "real_speed": "🛣️ Vitesse réelle courante (km/h)",
                "speed_diff": "📉 Écart (prédiction - réelle) (km/h)",
            }
            metric_sel = st.selectbox(
                "Variable à cartographier :", list(metric_options.keys()), format_func=lambda x: metric_options[x]
            )

        with f_col3:
            # Vitesse filter
            min_speed = float(df_preds["predicted_speed"].min())
            max_speed = float(df_preds["predicted_speed"].max())
            speed_range = st.slider(
                "Filtrer par vitesse prédite (km/h) :", min_speed, max_speed, (min_speed, max_speed)
            )

        # Filter the DataFrame
        df_filtered = df_preds[
            (df_preds["horizon_minutes"] == horizon_sel)
            & (df_preds["predicted_speed"] >= speed_range[0])
            & (df_preds["predicted_speed"] <= speed_range[1])
        ].copy()

        if not df_filtered.empty:
            # Key performance metrics for filtered subset
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Nombre de segments", len(df_filtered))
            with m_col2:
                st.metric("Vitesse moyenne prédite", f"{df_filtered['predicted_speed'].mean():.1f} km/h")
            with m_col3:
                st.metric("Vitesse moyenne réelle", f"{df_filtered['real_speed'].mean():.1f} km/h")
            with m_col4:
                avg_diff = df_filtered["speed_diff"].mean()
                diff_label = "Sur-estimation" if avg_diff > 0 else "Sous-estimation"
                st.metric(f"Biais moyen ({diff_label})", f"{avg_diff:+.2f} km/h")

            # 3. Plotly Mapbox Line Chart (Flatten LineStrings as proposed)
            import plotly.express as px
            import shapely.wkt

            # Define speed categories and color map
            speed_color_map = {
                "Lent (0-15 km/h)": "red",
                "Moyen (15-40 km/h)": "orange",
                "Rapide (>40 km/h)": "green",
                "Inconnu": "gray",
            }

            def get_speed_category(speed):
                if pd.isna(speed):
                    return "Inconnu"
                elif speed < 15:
                    return "Lent (0-15 km/h)"
                elif speed > 40:
                    return "Rapide (>40 km/h)"
                else:
                    return "Moyen (15-40 km/h)"

            # Define diff categories and color map for the speed difference
            diff_color_map = {
                "Sous-estimation (<-10 km/h)": "blue",
                "Précis ([-10, 10] km/h)": "green",
                "Sur-estimation (>10 km/h)": "red",
                "Inconnu": "gray",
            }

            def get_diff_category(diff):
                if pd.isna(diff):
                    return "Inconnu"
                elif diff <= -10:
                    return "Sous-estimation (<-10 km/h)"
                elif diff >= 10:
                    return "Sur-estimation (>10 km/h)"
                else:
                    return "Précis ([-10, 10] km/h)"

            # Expander for Map optimization options
            st.write("")
            with st.expander("⚙️ Options d'affichage de la carte (optimisation de la fluidité)", expanded=False):
                opt_col1, opt_col2, opt_col3 = st.columns(3)
                with opt_col1:
                    engine_sel = st.radio(
                        "Moteur de rendu cartographique :",
                        ["🚀 Streamlit natif (ultra-rapide, recommandé)", "🎨 Plotly Mapbox (interactif, secondaire)"],
                        index=0,
                        help="Sélectionnez 'Streamlit natif' pour un affichage ultra-fluide.",
                    )
                with opt_col2:
                    map_type_sel = st.radio(
                        "Style de représentation :",
                        ["🛣️ Lignes (tracés) - recommandé"],
                        index=0,
                        help="Affiche le tracé exact de chaque segment routier sous forme de ligne.",
                    )
                with opt_col3:
                    max_segments = st.slider(
                        "Nombre max de segments à tracer (mode lignes) :",
                        min_value=1000,
                        max_value=3000,
                        value=1400,
                        step=100,
                        help="Limite le nombre de lignes à dessiner pour éviter de figer le navigateur. Seuls les segments les plus critiques/congestionnés seront affichés.",
                    )

            # Re-prioritize and filter map data based on selection
            if "Points" in map_type_sel:
                df_map_data = df_filtered.copy()
            else:
                # Prioritize segments to draw
                if metric_sel == "speed_diff":
                    # Largest error differences first
                    df_map_data = (
                        df_filtered.sort_values(by="speed_diff", key=lambda x: x.abs(), ascending=False)
                        .head(max_segments)
                        .copy()
                    )
                elif metric_sel == "predicted_speed":
                    # Slowest predicted speed first
                    df_map_data = (
                        df_filtered.sort_values(by="predicted_speed", ascending=True).head(max_segments).copy()
                    )
                else:
                    # Slowest real speed first
                    df_map_data = df_filtered.sort_values(by="real_speed", ascending=True).head(max_segments).copy()

            # Assign category to the filtered DataFrame based on selected metric
            if metric_sel == "speed_diff":
                df_map_data["categorie"] = df_map_data["speed_diff"].apply(get_diff_category)
                color_map = diff_color_map
            else:
                df_map_data["categorie"] = df_map_data[metric_sel].apply(get_speed_category)
                color_map = speed_color_map

            # Define colors in Hex for native engine
            hex_color_map = {
                "Lent (0-15 km/h)": "#E11D48",  # Beautiful red (Tailwind rose-600)
                "Moyen (15-40 km/h)": "#F59E0B",  # Beautiful orange (Tailwind amber-500)
                "Rapide (>40 km/h)": "#10B981",  # Beautiful green (Tailwind emerald-500)
                "Inconnu": "#6B7280",  # Gray (Tailwind gray-500)
                "Sous-estimation (<-10 km/h)": "#3B82F6",  # Beautiful blue (Tailwind blue-500)
                "Précis ([-10, 10] km/h)": "#10B981",  # Beautiful green (Tailwind emerald-500)
                "Sur-estimation (>10 km/h)": "#EF4444",  # Beautiful red (Tailwind red-500)
            }

            def hex_to_rgb(h_str):
                h_str = h_str.lstrip("#")
                return list(int(h_str[i : i + 2], 16) for i in (0, 2, 4))

            # Render the Map in a full-width container to display a stunning large responsive square map
            col_map_mid = st.container()
            with col_map_mid:
                st.markdown(
                    """
                    <div id="map-section"></div>
                    """,
                    unsafe_allow_html=True,
                )

                if "Streamlit Natif" in engine_sel or "Streamlit natif" in engine_sel or "natif" in engine_sel.lower():
                    # Add color hex column to df_map_data
                    df_map_data["color"] = df_map_data["categorie"].map(hex_color_map).fillna("#6B7280")

                    if "Points" in map_type_sel:
                        log_time("Rendering points using st.map...")
                        st.map(
                            df_map_data, latitude="latitude", longitude="longitude", color="color", size=15, zoom=11.5
                        )
                        log_time("Points rendered")
                    else:
                        # Render lines with pydeck PathLayer
                        log_time("Building PyDeck path layer data...")
                        import pydeck as pdk

                        path_data = []
                        for idx, row in df_map_data.iterrows():
                            coords = row.get("parsed_path_coords")
                            if coords:
                                path_data.append(
                                    {
                                        "path": coords,
                                        "color": hex_to_rgb(row["color"]),
                                        "name": str(row["nom_rue"]),
                                        "val": float(row[metric_sel]),
                                        "val_str": f"{float(row[metric_sel]):.1f}",
                                    }
                                )
                        log_time(f"PyDeck path data built (count={len(path_data)})")

                        if path_data:
                            layer = pdk.Layer(
                                "PathLayer",
                                path_data,
                                get_path="path",
                                get_color="color",
                                get_width=5,
                                width_min_pixels=3,
                                pickable=True,
                            )
                            log_time("Calling st.pydeck_chart...")
                            st.pydeck_chart(
                                pdk.Deck(
                                    layers=[layer],
                                    initial_view_state=pdk.ViewState(
                                        latitude=45.764043, longitude=4.835659, zoom=11.5, pitch=0
                                    ),
                                    map_style="light",
                                    width=600,
                                    height=600,
                                    tooltip={
                                        "html": "<b>Rue :</b> {name}<br/><b>Vitesse :</b> {val_str} km/h",
                                        "style": {
                                            "backgroundColor": "#1E293B",
                                            "color": "white",
                                            "fontFamily": "'Outfit', 'Inter', Arial, sans-serif",
                                            "fontSize": "13px",
                                            "padding": "10px",
                                            "borderRadius": "6px",
                                            "border": "1px solid #334155",
                                        },
                                    },
                                ),
                                use_container_width=True,
                            )
                            log_time("st.pydeck_chart call complete")
                        else:
                            st.error(
                                "❌ Erreur de génération géométrique : Impossible d'extraire les coordonnées des segments routiers."
                            )

                    # Display elegant legend below the map
                    if metric_sel == "speed_diff":
                        st.markdown(
                            """
                            <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 20px; padding: 12px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e5e7eb;">
                                <span style="font-weight: bold; font-size: 14px; color: #374151;">Légende des couleurs :</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #EF4444; border-radius: 50%; display: inline-block;"></span> Sur-estimation (&gt;10 km/h)</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span> Conforme ([-10, 10] km/h)</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #3B82F6; border-radius: 50%; display: inline-block;"></span> Sous-estimation (&lt;-10 km/h)</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #6B7280; border-radius: 50%; display: inline-block;"></span> Inconnu</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 10px; margin-bottom: 20px; padding: 12px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e5e7eb;">
                                <span style="font-weight: bold; font-size: 14px; color: #374151;">Légende des couleurs :</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #E11D48; border-radius: 50%; display: inline-block;"></span> Lent (&lt;15 km/h)</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #F59E0B; border-radius: 50%; display: inline-block;"></span> Moyen (15-40 km/h)</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span> Rapide (&gt;40 km/h)</span>
                                <span style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: #4B5563;"><span style="height: 12px; width: 12px; background-color: #6B7280; border-radius: 50%; display: inline-block;"></span> Inconnu</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:
                    # Render with Plotly Mapbox
                    if "Points" not in map_type_sel and len(df_map_data) > 500:
                        st.warning(
                            "⚠️ **Attention :** Le rendu de plus de 500 tracés de lignes avec Plotly Mapbox peut fortement ralentir ou figer le navigateur. Pour une fluidité maximale, nous vous recommandons d'utiliser le moteur **Streamlit natif** (recommandé et actif par défaut)."
                        )
                    if "Points" in map_type_sel:
                        fig_map = px.scatter_mapbox(
                            df_map_data,
                            lat="latitude",
                            lon="longitude",
                            color="categorie",
                            color_discrete_map=color_map,
                            hover_name="nom_rue",
                            hover_data={
                                "properties_twgid": True,
                                "latitude": False,
                                "longitude": False,
                                metric_sel: True,
                                "categorie": False,
                            },
                            labels={
                                "nom_rue": "Rue / Segment",
                                "properties_twgid": "Identifiant (twgid)",
                                metric_sel: "Vitesse (km/h)",
                            },
                            mapbox_style="carto-positron",
                            zoom=11.5,
                            center={"lat": 45.764043, "lon": 4.835659},  # Centré sur Lyon
                            height=600,
                            title=f"État du réseau routier ({len(df_map_data)} segments affichés en mode point)",
                        )
                        fig_map.update_traces(marker=dict(size=9, opacity=0.85))

                        # Design tweaks
                        fig_map.update_layout(
                            width=600,
                            height=600,
                            margin={"r": 0, "t": 30, "l": 0, "b": 0},
                            legend=dict(
                                title="Catégorie",
                                yanchor="top",
                                y=0.95,
                                xanchor="left",
                                x=0.02,
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="#E5E7EB",
                                borderwidth=1,
                            ),
                        )
                        st.plotly_chart(fig_map, use_container_width=True)

                    else:
                        lats = []
                        lons = []
                        noms = []
                        vitesses = []
                        categories = []
                        ids_rue = []

                        for idx, row in df_map_data.iterrows():
                            coords = row.get("parsed_path_coords")
                            if coords:
                                for lon, lat in coords:
                                    lons.append(lon)
                                    lats.append(lat)
                                    noms.append(row.get("nom_rue", f"Segment {row['properties_twgid']}"))
                                    vitesses.append(row[metric_sel])
                                    categories.append(row["categorie"])
                                    # Use the unique index of the row to ensure each segment is treated as a separate trace/line group
                                    ids_rue.append(idx)

                        df_lines = pd.DataFrame(
                            {
                                "lat": lats,
                                "lon": lons,
                                "nom": noms,
                                "vitesse": vitesses,
                                "categorie": categories,
                                "id_rue": ids_rue,
                            }
                        )

                        if not df_lines.empty:
                            fig_map = px.line_mapbox(
                                df_lines,
                                lat="lat",
                                lon="lon",
                                line_group="id_rue",  # Relie les points de la même rue, sans relier les rues entre elles
                                color="categorie",  # Catégorie de trafic / vitesse
                                color_discrete_map=color_map,
                                hover_name="nom",
                                hover_data={
                                    "lat": False,
                                    "lon": False,
                                    "vitesse": True,
                                    "id_rue": True,
                                    "categorie": False,
                                },
                                labels={
                                    "nom": "Rue / Segment",
                                    "id_rue": "Identifiant de segment",
                                    "vitesse": "Vitesse (km/h)",
                                },
                                mapbox_style="carto-positron",
                                zoom=11.5,
                                center={"lat": 45.764043, "lon": 4.835659},  # Centré sur Lyon
                                height=600,
                                title=f"État du réseau routier (tracés des {len(df_map_data)} segments les plus critiques)",
                            )

                            # Map design tweaks
                            fig_map.update_layout(
                                width=600,
                                height=600,
                                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                                legend=dict(
                                    title="Catégorie",
                                    yanchor="top",
                                    y=0.95,
                                    xanchor="left",
                                    x=0.02,
                                    bgcolor="rgba(255, 255, 255, 0.8)",
                                    bordercolor="#E5E7EB",
                                    borderwidth=1,
                                ),
                            )
                            # Ensure lines are bold, thick, and highly visible on the map (default is 1-2, make it 5)
                            fig_map.update_traces(line=dict(width=5))

                            st.plotly_chart(fig_map, use_container_width=True)
                            st.caption(
                                f"🛣️ Rendu de la carte en mode **lignes (tracés)** : **{df_lines['id_rue'].nunique()}** segments routiers dessinés via un tracé continu de **{len(df_lines)}** coordonnées géométriques."
                            )
                        else:
                            st.error(
                                "❌ Erreur de génération géométrique : Impossible d'extraire les coordonnées des segments routiers."
                            )

            # Model Reference card shown below the map
            st.markdown(
                f"""
                <div class="metric-card" style="max-width: 450px; margin: 30px auto 0 auto; text-align: center; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <h3 style="font-size: 1.1rem; color: #4b5563; margin-top: 0; margin-bottom: 5px;">Modèle de prédiction actif</h3>
                    <h2 style="color: #10B981; font-size: 1.8rem; margin: 5px 0;">{selected_run_model}</h2>
                    <p style="font-size: 0.9rem; color: #6b7280; font-family: monospace; background-color: #f8fafc; padding: 6px 12px; border-radius: 6px; display: inline-block; margin-bottom: 0;">{selected_run_name}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.warning("⚠️ Aucun segment ne correspond aux filtres de vitesse sélectionnés.")
    else:
        st.info(
            "🗺️ Les données d'ingestion et de prédiction s'afficheront ici dès que le premier cycle d'orchestration Airflow aura complété l'ingestion bronze et la transformation silver."
        )
        st.image(
            "https://raw.githubusercontent.com/evidentlyai/evidently/main/docs/book/_static/evidently_logo.png",
            width=300,
        )
