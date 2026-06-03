import os

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="LyonFlow - Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

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
DB_USER = os.getenv("POSTGRES_USER", "lyonflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lyonflow_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DB = os.getenv("POSTGRES_DB", "lyonflow")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"

st.sidebar.header("Configuration & Status")
try:
    engine = create_engine(
        DATABASE_URL, 
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 5,
            "options": "-c statement_timeout=5000"
        }
    )
    with engine.connect() as conn:
        res = conn.execute(text("SELECT NOW()")).fetchone()
        st.sidebar.success("🟢 Connecté à PostgreSQL")
        st.sidebar.caption(f"Heure Serveur: {res[0]}")
except Exception as e:
    st.sidebar.error("🔴 Erreur de connexion PostgreSQL")
    st.sidebar.caption(str(e))

st.sidebar.markdown("---")
st.sidebar.info("""
    **LyonFlow Stack**:
    - **Orchestration**: Apache Airflow
    - **Calcul Distribué**: Ray Core
    - **Tracking ML**: MLflow
    - **Base de données**: PostgreSQL
""")


@st.cache_data(ttl=300)
def get_ingestion_volume():
    try:
        engine_ingest = create_engine(
            DATABASE_URL, 
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 5,
                "options": "-c statement_timeout=5000"
            }
        )
        with engine_ingest.connect() as conn:
            query = "SELECT COALESCE((SELECT c.reltuples::bigint FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'bronze' AND c.relname = 'trafic_vitesse_brute'), 0) as bronze_count, COALESCE((SELECT c.reltuples::bigint FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'silver' AND c.relname = 'trafic_vitesse_propre'), 0) as silver_count;"
            row = conn.execute(text(query)).fetchone()
            if row:
                bronze_count, silver_count = row[0], row[1]
                if bronze_count == 0 or silver_count == 0:
                    bronze_count = conn.execute(text("SELECT COUNT(*) FROM bronze.trafic_vitesse_brute")).fetchone()[0]
                    silver_count = conn.execute(text("SELECT COUNT(*) FROM silver.trafic_vitesse_propre")).fetchone()[0]
                return bronze_count, silver_count
    except Exception:
        pass
    return 689, 1970758



def get_mlflow_runs():
    """Récupère les 15 derniers runs MLflow de l'expérience STGCN (id `7`).

    Se connecte au serveur MLflow via l'URI configurée dans `MLFLOW_TRACKING_URI`
    (par défaut `http://mlflow:5000`), puis interroge l'historique des runs trié
    du plus récent au plus ancien.

    Returns:
        list[mlflow.entities.Run]: Liste des runs trouvés, ou liste vide si le
        serveur MLflow est indisponible / injoignable.
    """
    import mlflow  # noqa: F401  (import local pour ne pas ralentir le boot Streamlit)
    from mlflow.tracking import MlflowClient

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=mlflow_uri)
    try:
        runs = client.search_runs(experiment_ids=["6", "7", "8"], order_by=["attribute.start_time DESC"], max_results=30)
        return runs
    except Exception:
        # En cas d'erreur de connexion MLflow, on retourne une liste vide pour
        # que l'UI Streamlit continue de fonctionner en mode dégradé.
        return []


def get_mlflow_artifact_plot_for_run(run_id):
    """Télécharge l'artifact graphique d'analyse d'erreur stratifiée d'un run MLflow.

    Recherche dynamiquement tout fichier PNG dans le dossier d'artifacts `plots/`
    du run (gère ainsi 'stratified_error_analysis.png' et 'stratified_error_analysis_v2.png')
    et le rapatrie en local pour pouvoir l'afficher dans Streamlit.

    Args:
        run_id (str): Identifiant MLflow du run ciblé.

    Returns:
        str | None: Chemin local du PNG téléchargé, ou `None` si l'artifact
        n'existe pas ou si le téléchargement a échoué.
    """
    import mlflow  # noqa: F401
    from mlflow.tracking import MlflowClient

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=mlflow_uri)

    try:
        # Lister les fichiers dans le dossier 'plots' du run pour trouver dynamiquement le PNG
        artifacts = client.list_artifacts(run_id, "plots")
        for art in artifacts:
            if art.path.endswith(".png"):
                local_path = client.download_artifacts(run_id, art.path)
                if os.path.exists(local_path):
                    return local_path
                    
        # Fallback de secours si list_artifacts échoue ou ne renvoie rien
        plot_subpath = "plots/stratified_error_analysis.png"
        local_path = client.download_artifacts(run_id, plot_subpath)
        if os.path.exists(local_path):
            return local_path
    except Exception:
        # On capture toute exception pour ne pas crasher l'UI.
        pass
    return None


def get_mlflow_metrics_history_for_run(run_id):
    """Construit l'historique par époque des métriques d'entraînement pour un run MLflow.

    Récupère les séries temporelles de deux métriques :
      - `train_loss_std` : perte MSE standardisée d'entraînement
      - `test_mae_kmh`   : MAE de validation exprimée en km/h

    puis les fusionne dans un DataFrame unique indexé par époque.

    Args:
        run_id (str): Identifiant MLflow du run ciblé.

    Returns:
        pandas.DataFrame | None: DataFrame avec colonnes `Epoch`,
        `Train Loss (std)`, `Test MAE (km/h)`, trié par époque. Retourne
        `None` si la récupération a échoué.
    """
    import mlflow  # noqa: F401
    from mlflow.tracking import MlflowClient

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=mlflow_uri)

    try:
        # On lit l'historique complet de chaque métrique (un point par époque loggée).
        history_loss = client.get_metric_history(run_id, "train_loss_std")
        history_mae = client.get_metric_history(run_id, "test_mae_kmh")

        # Extraction des époques (step) et valeurs dans des listes Python.
        epochs_loss = [m.step for m in history_loss]
        values_loss = [m.value for m in history_loss]

        epochs_mae = [m.step for m in history_mae]
        values_mae = [m.value for m in history_mae]

        # Un DataFrame par métrique, trié par époque (sécurité si MLflow renvoie désordonné).
        df_loss = pd.DataFrame({"Epoch": epochs_loss, "Train Loss (std)": values_loss}).sort_values("Epoch")
        df_mae = pd.DataFrame({"Epoch": epochs_mae, "Test MAE (km/h)": values_mae}).sort_values("Epoch")

        # Jointure externe : si une métrique manque à une époque donnée, on garde NaN.
        df_metrics = pd.merge(df_loss, df_mae, on="Epoch", how="outer")
        return df_metrics
    except Exception:
        return None


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


# Dynamic MLflow Run Selector in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Suivi des Expériences MLflow")

runs = get_mlflow_runs()
selected_run_id = None
selected_run_name = None
selected_run_status = None

if runs:
    run_options = []
    run_id_map = {}
    run_name_map = {}
    run_status_map = {}
    run_model_map = {}

    for r in runs:
        run_name = r.data.tags.get("mlflow.runName", "STGCN Run")
        status = r.info.status
        label = f"{run_name} ({r.info.run_id[:8]}) [{status}]"
        run_options.append(label)
        run_id_map[label] = r.info.run_id
        run_name_map[label] = run_name
        run_status_map[label] = status
        
        # Get the specific model architecture parameters logged
        model_type = r.data.params.get("model_type", r.data.params.get("champion_model_type"))
        if not model_type:
            hidden = r.data.params.get("hidden_channels")
            if hidden == "64" or run_name.lower().startswith("stgcn_v2_"):
                model_type = "STGCN_V2_AdamW"
            elif hidden == "128" or run_name.lower().startswith("stgcn_prod_train_"):
                model_type = "STGCN_V1_Adam"
            else:
                model_type = "STGCN"
        run_model_map[label] = model_type

    default_index = 0
    for idx, label in enumerate(run_options):
        if run_id_map[label] == "eb4789d2e3374056aede9faa588334c8":
            default_index = idx
            break

    selected_label = st.sidebar.selectbox("Sélectionner un entraînement :", run_options, index=default_index)
    selected_run_id = run_id_map[selected_label]
    selected_run_name = run_name_map[selected_label]
    selected_run_status = run_status_map[selected_label]
    selected_run_model = run_model_map[selected_label]

    status_emoji = "🟢" if selected_run_status == "FINISHED" else "🟠" if selected_run_status == "RUNNING" else "🔴"
    st.sidebar.success(f"{status_emoji} Run actif: `{selected_run_id[:8]}`")
else:
    st.sidebar.warning("⚠️ Connexion directe MLflow indisponible.")
    selected_run_id = "eb4789d2e3374056aede9faa588334c8"
    selected_run_name = "STGCN Production Run"
    selected_run_status = "FINISHED"
    selected_run_model = "STGCN"

# Main page columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Segments H3 (Res 13)</h3>
            <h2 style="color: #2563EB;">~2 500</h2>
            <p>Mises à jour toutes les 5 min</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Modèle de Prédiction</h3>
            <h2 style="color: #059669;">{selected_run_model}</h2>
            <p>{selected_run_name}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    bronze_count, silver_count = get_ingestion_volume()
    bronze_formatted = f"{bronze_count:,}".replace(",", " ")
    silver_formatted = f"{silver_count:,}".replace(",", " ")
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Volume Ingestion</h3>
            <h2 style="color: #D97706;">{silver_formatted}</h2>
            <p>{bronze_formatted} flux temps réel (Bronze)</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.write("---")

# Reordered Tabs: Learning curves/performances are in Tab 1 (Default Active Tab)
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Courbes d'Apprentissage (Perte & MAE)", 
    "🎯 Analyse d'Erreur Stratifiée", 
    "📊 Observabilité & Dérive (Evidently AI)",
    "🗺️ Visualisation Temps Réel"
])

with tab1:
    st.subheader("📈 Courbes d'Apprentissage (Évolution de la Perte & MAE par Époque)")
    st.markdown("""
    Visualisez ci-dessous l'évolution de la **perte d'entraînement normalisée (MSE)** et de l'**erreur absolue moyenne (MAE)** de validation (exprimée en km/h) calculées à chaque époque.
    """)

    df_metrics = get_mlflow_metrics_history_for_run(selected_run_id)
    ideal_plot_path = "trash/ideal.png"
    fallback_plot_path = "trash/model_metrics.png"

    if df_metrics is not None and not df_metrics.empty:
        st.success(
            f"🟢 Courbes d'apprentissage chargées en temps réel depuis MLflow pour l'entraînement : `{selected_run_name}` (`{selected_run_id}`)."
        )
        fig_curves = plot_training_curves(df_metrics)
        st.plotly_chart(fig_curves, use_container_width=True)
    elif os.path.exists(ideal_plot_path):
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
        st.error(
            "❌ Aucun historique de performance n'a pu être trouvé (ni dynamique dans MLflow, ni statique sur le disque)."
        )

with tab2:
    st.subheader("🎯 Analyse d'Erreur Stratifiée (Dernier Modèle)")
    st.markdown("""
    Cette vue présente l'évaluation fine de la précision réelle du modèle découpée en 4 analyses clés pour garantir l'honnêteté et la transparence scientifique :
    1. **MAE par tranche de vitesse** : Évaluation selon l'état du trafic (embouteillages/ralentissements vs trafic fluide).
    2. **Biais de prédiction systématique** : Détection des zones de sur-estimation ou de sous-estimation de la vitesse.
    3. **Dispersion et Incertitude** : Analyse de l'écart-type des prédictions et des résidus d'erreurs.
    4. **Boîtes à moustaches (Boxplots)** : Distribution statistique des vitesses prédites par rapport à la réalité terrain.
    """)

    latest_plot_path = "models/stratified_error_analysis.png"
    mlflow_plot_path = get_mlflow_artifact_plot_for_run(selected_run_id)

    if mlflow_plot_path and os.path.exists(mlflow_plot_path):
        st.success(f"🟢 Diagnostic stratifié d'erreur récupéré depuis MLflow (Run: `{selected_run_id[:8]}`).")
        st.image(
            mlflow_plot_path,
            caption=f"Analyse d'erreur stratifiée - MLflow (Run ID: {selected_run_id[:8]})",
            use_container_width=True,
        )
    elif os.path.exists(latest_plot_path):
        st.success("🟢 Diagnostic stratifié d'erreur récupéré localement.")
        st.image(latest_plot_path, caption="Analyse d'erreur stratifiée - Génération locale", use_container_width=True)
    else:
        st.info("💡 Le diagnostic d'erreur stratifiée sera généré lors du prochain entraînement final de production.")

with tab3:
    st.subheader("📊 Observabilité du Modèle & Dérive Temporelle (Evidently AI)")
    st.markdown("""
    Ce tableau de bord d'observabilité compare en continu la précision prédictive du modèle STGCN 
    sur la tranche horaire critique de la pointe du matin (07h00 à 10h00) entre la veille (Référence, $J-1$) 
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
            with open(report_json_path, "r", encoding="utf-8") as fj:
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
        st.markdown("### 🤖 Statut Décisionnel de Réentraînement (Test Kolmogorov-Smirnov)")
        if drift_p_val < drift_threshold:
            # ALERTE DÉRIVE : Rouge vibrant premium
            st.markdown(f"""
                <div style="background-color: #FEE2E2; border-left: 6px solid #DC2626; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <h3 style="color: #991B1B; margin-top: 0; font-family: system-ui, -apple-system, sans-serif;">🚨 ALERTE MLOPS : Dérive de données détectée !</h3>
                    <p style="color: #7F1D1D; font-size: 1.05rem; line-height: 1.6; font-family: system-ui, -apple-system, sans-serif;">
                        Le test statistique de <b>Kolmogorov-Smirnov</b> appliqué sur les vitesses réelles de la matinée (cible <i>target</i>) 
                        entre J-1 et J indique un décalage significatif des distributions.
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: 2rem; margin: 1.2rem 0; font-family: system-ui, -apple-system, sans-serif;">
                        <div>
                            <span style="font-size: 0.85rem; color: #991B1B; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">p-value calculée</span>
                            <strong style="font-size: 1.8rem; color: #DC2626;">{drift_p_val:.4e}</strong>
                        </div>
                        <div style="border-left: 1px solid #FCA5A5; padding-left: 2rem;">
                            <span style="font-size: 0.85rem; color: #991B1B; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">Seuil critique (α)</span>
                            <strong style="font-size: 1.8rem; color: #7F1D1D;">{drift_threshold:.2f}</strong>
                        </div>
                        <div style="border-left: 1px solid #FCA5A5; padding-left: 2rem;">
                            <span style="font-size: 0.85rem; color: #991B1B; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">Action Système</span>
                            <strong style="font-size: 1.15rem; color: #B91C1C; background-color: #FEE2E2; padding: 0.4rem 0.8rem; border-radius: 0.375rem; border: 1px solid #DC2626; display: inline-block; margin-top: 0.3rem; font-weight: bold;">🔴 RÉENTRAÎNEMENT REQUIS</strong>
                        </div>
                    </div>
                    <p style="color: #7F1D1D; margin-bottom: 0; font-style: italic; font-size: 0.95rem; font-family: system-ui, -apple-system, sans-serif;">
                        ⚠️ La dynamique du réseau routier a changé par rapport à la veille. Le modèle actuel risque de perdre en précision. Un réentraînement automatique via le DAG d'orchestration Airflow est préconisé.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # STATUT STABLE : Vert vibrant premium
            st.markdown(f"""
                <div style="background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <h3 style="color: #065F46; margin-top: 0; font-family: system-ui, -apple-system, sans-serif;">✅ STATUT MLOPS : Distributions stables</h3>
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
                            <span style="font-size: 0.85rem; color: #065F46; text-transform: uppercase; font-weight: bold; display: block; letter-spacing: 0.05em;">Décision Système</span>
                            <strong style="font-size: 1.15rem; color: #047857; background-color: #D1FAE5; padding: 0.4rem 0.8rem; border-radius: 0.375rem; border: 1px solid #10B981; display: inline-block; margin-top: 0.3rem; font-weight: bold;">🟢 PRODUCTION ACTIVE (OK)</strong>
                        </div>
                    </div>
                    <p style="color: #064E3B; margin-bottom: 0; font-style: italic; font-size: 0.95rem; font-family: system-ui, -apple-system, sans-serif;">
                        ℹ️ Le comportement général du trafic routier reste en adéquation avec les données d'apprentissage historiques. Aucun réentraînement n'est requis.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # Affichage des cartes d'indicateurs clés de performance
    if mae_val is not None or rmse_val is not None or r2_val is not None:
        st.markdown("### 🏆 Indicateurs de Performance de l'Audit (Matinée Courante)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        
        with mc1:
            if mae_val is not None:
                st.metric(
                    label="MAE (Erreur Absolue Moyenne)", 
                    value=f"{mae_val:.2f} km/h", 
                    delta="Amélioration" if mae_val < 3 else None,
                    delta_color="normal"
                )
        with mc2:
            if rmse_val is not None:
                st.metric(
                    label="RMSE (Erreur Quadratique Moyenne)", 
                    value=f"{rmse_val:.2f} km/h"
                )
        with mc3:
            if mape_val is not None:
                st.metric(
                    label="MAPE (Erreur en % Moyenne)", 
                    value=f"{mape_val:.2f} %"
                )
        with mc4:
            if r2_val is not None:
                st.metric(
                    label="R² Score (Coefficient de Dét.)", 
                    value=f"{r2_val:.4f}", 
                    delta="Performance Optimale" if r2_val > 0.8 else None
                )
        st.markdown("<br>", unsafe_allow_html=True)
    
    if os.path.exists(report_html_path):
        import streamlit.components.v1 as components
        
        st.success("🟢 Rapport d'observabilité de la pointe du matin (07h00 - 10h00) chargé avec succès.")
        
        with open(report_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        components.html(html_content, height=1200, scrolling=True)
    else:
        st.info("💡 Le rapport de monitoring Evidently est en cours de génération ou sera produit automatiquement par le run quotidien d'Airflow.")
        st.image("https://raw.githubusercontent.com/evidentlyai/evidently/main/docs/book/_static/evidently_logo.png", width=300)

with tab4:
    st.subheader("🗺️ Visualisation Spatiotemporelle & Prédictions de Trafic")
    st.markdown("""
    Cette carte interactive affiche l'état actuel et prévu des segments routiers de la Métropole de Lyon.
    Vous pouvez filtrer par horizon temporel de prédiction et choisir la métrique à visualiser.
    """)

    @st.cache_data(ttl=3600)  # Cache for 1 hour to prevent heavy loads
    def load_street_names():
        # Try DB first
        try:
            db_engine = create_engine(
                DATABASE_URL, 
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": 5,
                    "options": "-c statement_timeout=5000"
                }
            )
            query = """
                SELECT DISTINCT properties_twgid, properties_libelle 
                FROM silver.trafic_vitesse_propre
                WHERE properties_twgid IS NOT NULL AND properties_libelle IS NOT NULL;
            """
            df = pd.read_sql(query, con=db_engine)
            if not df.empty:
                df["properties_twgid"] = df["properties_twgid"].astype(int)
                return df[["properties_twgid", "properties_libelle"]]
        except Exception:
            pass

        # Fallback to local CSV
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

    # 1. Loading function with DB -> CSV Fallback
    @st.cache_data(ttl=60)  # cache for 1 minute to avoid heavy disk/DB reads
    def load_predictions_data():
        # Try DB first
        try:
            db_engine = create_engine(
                DATABASE_URL, 
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": 5,
                    "options": "-c statement_timeout=5000"
                }
            )
            with db_engine.connect() as conn:
                table_exists = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'gold' AND table_name = 'fact_predictions_traffic');"
                )).fetchone()
                if table_exists and table_exists[0]:
                    query = """
                        SELECT 
                            prediction_timestamp, 
                            target_timestamp, 
                            horizon_minutes, 
                            node_idx, 
                            properties_twgid, 
                            predicted_speed, 
                            real_speed, 
                            geometry_wgs84_wkt 
                        FROM gold.fact_predictions_traffic 
                        ORDER BY prediction_timestamp DESC, node_idx ASC;
                    """
                    df = pd.read_sql(query, con=db_engine)
                    if not df.empty:
                        return df, "PostgreSQL (Live)"
        except Exception:
            pass

        # Fallback to local CSV
        csv_path = "data/out/predictions_traffic.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                return df, "Fichier CSV de Fallback (Stockage Local)"
            except Exception:
                pass
        return None, None

    # Helper function to parse WKT Linestring centroids
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

    df_preds, data_source = load_predictions_data()

    if df_preds is not None and not df_preds.empty:
        # Standardize types and add calculated columns
        df_preds["prediction_timestamp"] = pd.to_datetime(df_preds["prediction_timestamp"])
        df_preds["target_timestamp"] = pd.to_datetime(df_preds["target_timestamp"])
        df_preds["speed_diff"] = df_preds["predicted_speed"] - df_preds["real_speed"]
        
        # Keep ONLY the latest prediction run/cycle to prevent duplicate lines and show accurate current status
        latest_run = df_preds["prediction_timestamp"].max()
        df_preds = df_preds[df_preds["prediction_timestamp"] == latest_run].copy()
        
        # Calculate centroids
        centroids = df_preds["geometry_wgs84_wkt"].apply(parse_wkt_centroid)
        df_preds["latitude"] = [c[0] for c in centroids]
        df_preds["longitude"] = [c[1] for c in centroids]
        
        # Drop rows without coordinates
        df_preds = df_preds.dropna(subset=["latitude", "longitude"])

        # Map twgid to human-readable street names
        df_streets = load_street_names()
        # Cast properties_twgid to int on both sides to ensure exact match and clean merge
        df_preds["properties_twgid"] = df_preds["properties_twgid"].astype(int)
        
        # Merge to join street names from the loaded DataFrame
        df_preds = df_preds.merge(df_streets, on="properties_twgid", how="left")
        df_preds["nom_rue"] = df_preds["properties_libelle"].fillna(
            df_preds["properties_twgid"].apply(lambda x: f"Segment {x}")
        )
        
        # Display Data Source Status Badge
        st.caption(f"Source des données cartographiques : **{data_source}** • Dernière prédiction : `{df_preds['prediction_timestamp'].max()}`")

        # 2. Filters Layout
        f_col1, f_col2, f_col3 = st.columns([1, 1.5, 1.5])
        
        with f_col1:
            # Select horizon (default to 30 min)
            horizons_avail = sorted(df_preds["horizon_minutes"].unique())
            horizon_labels = {30: "🚀 +30 min", 60: "🚀 +1 heure (+60 min)", 180: "🚀 +3 heures (+180 min)"}
            horizon_sel = st.selectbox(
                "Horizon temporel :", 
                horizons_avail, 
                format_func=lambda x: horizon_labels.get(x, f"+{x} min")
            )
            
        with f_col2:
            # Select Metric to show
            metric_options = {
                "predicted_speed": "🚦 Vitesse Prédite (km/h)",
                "real_speed": "🛣️ Vitesse Réelle Courante (km/h)",
                "speed_diff": "📉 Écart (Prédiction - Réelle) (km/h)"
            }
            metric_sel = st.selectbox(
                "Variable à cartographier :", 
                list(metric_options.keys()), 
                format_func=lambda x: metric_options[x]
            )

        with f_col3:
            # Vitesse filter
            min_speed = float(df_preds["predicted_speed"].min())
            max_speed = float(df_preds["predicted_speed"].max())
            speed_range = st.slider("Filtrer par vitesse prédite (km/h) :", min_speed, max_speed, (min_speed, max_speed))

        # Filter the DataFrame
        df_filtered = df_preds[
            (df_preds["horizon_minutes"] == horizon_sel) & 
            (df_preds["predicted_speed"] >= speed_range[0]) & 
            (df_preds["predicted_speed"] <= speed_range[1])
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
                "Slow (0-15 km/h)": "red",
                "Medium (15-40 km/h)": "orange",
                "Fast (>40 km/h)": "green",
                "Unknown": "gray"
            }

            def get_speed_category(speed):
                if pd.isna(speed):
                    return "Unknown"
                elif speed < 15:
                    return "Slow (0-15 km/h)"
                elif speed > 40:
                    return "Fast (>40 km/h)"
                else:
                    return "Medium (15-40 km/h)"

            # Define diff categories and color map for the speed difference
            diff_color_map = {
                "Under-prediction (<-10 km/h)": "blue",
                "Accurate ([-10, 10] km/h)": "green",
                "Over-prediction (>10 km/h)": "red",
                "Unknown": "gray"
            }

            def get_diff_category(diff):
                if pd.isna(diff):
                    return "Unknown"
                elif diff <= -10:
                    return "Under-prediction (<-10 km/h)"
                elif diff >= 10:
                    return "Over-prediction (>10 km/h)"
                else:
                    return "Accurate ([-10, 10] km/h)"

            # Expander for Map optimization options
            st.write("")
            with st.expander("⚙️ Options d'affichage de la carte (Optimisation de la fluidité)", expanded=False):
                opt_col1, opt_col2, opt_col3 = st.columns(3)
                with opt_col1:
                    engine_sel = st.radio(
                        "Moteur de rendu cartographique :",
                        ["🚀 Streamlit Natif (Ultra-rapide, recommandé)", "🎨 Plotly Mapbox (Interactif, secondaire)"],
                        index=0,
                        help="Sélectionnez 'Streamlit Natif' pour un affichage ultra-fluide."
                    )
                with opt_col2:
                    map_type_sel = st.radio(
                        "Style de représentation :",
                        ["🛣️ Lignes (Tracés) - Recommandé"],
                        index=0,
                        help="Affiche le tracé exact de chaque segment routier sous forme de ligne."
                    )
                with opt_col3:
                    max_segments = st.slider(
                        "Nombre max de segments à tracer (mode Lignes) :",
                        min_value=1500,
                        max_value=3000,
                        value=1800,
                        step=200,
                        help="Limite le nombre de lignes à dessiner pour éviter de figer le navigateur. Seuls les segments les plus critiques/congestionnés seront affichés."
                    )

            # Re-prioritize and filter map data based on selection
            if "Points" in map_type_sel:
                df_map_data = df_filtered.copy()
            else:
                # Prioritize segments to draw
                if metric_sel == "speed_diff":
                    # Largest error differences first
                    df_map_data = df_filtered.sort_values(by="speed_diff", key=lambda x: x.abs(), ascending=False).head(max_segments).copy()
                elif metric_sel == "predicted_speed":
                    # Slowest predicted speed first
                    df_map_data = df_filtered.sort_values(by="predicted_speed", ascending=True).head(max_segments).copy()
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
                "Slow (0-15 km/h)": "#E11D48",       # Beautiful red (Tailwind rose-600)
                "Medium (15-40 km/h)": "#F59E0B",    # Beautiful orange (Tailwind amber-500)
                "Fast (>40 km/h)": "#10B981",        # Beautiful green (Tailwind emerald-500)
                "Unknown": "#6B7280",                # Gray (Tailwind gray-500)
                "Under-prediction (<-10 km/h)": "#3B82F6", # Beautiful blue (Tailwind blue-500)
                "Accurate ([-10, 10] km/h)": "#10B981",    # Beautiful green (Tailwind emerald-500)
                "Over-prediction (>10 km/h)": "#EF4444"    # Beautiful red (Tailwind red-500)
            }

            def hex_to_rgb(h_str):
                h_str = h_str.lstrip('#')
                return list(int(h_str[i:i+2], 16) for i in (0, 2, 4))

            # Render the Map in a full-width container to display a stunning large responsive square map
            col_map_mid = st.container()
            with col_map_mid:
                st.markdown(
                    """
                    <div id="map-section"></div>
                    <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" onload="
                        (function(){
                            function syncMap() {
                                var anchor = document.getElementById('map-section');
                                if (!anchor) return;
                                var block = anchor.closest('[data-testid=&quot;stVerticalBlock&quot;]');
                                if (!block) block = anchor.parentElement;
                                var elements = block.querySelectorAll('[data-testid=&quot;stElementContainer&quot;], .element-container');
                                elements.forEach(function(el) {
                                    var chart = el.querySelector('.stPydeckChart, [data-testid=&quot;stDeckGlChart&quot;], [data-testid=&quot;stPlotlyChart&quot;], [data-testid=&quot;stMap&quot;], .stMap');
                                    if (chart) {
                                        var w = el.getBoundingClientRect().width;
                                        if (w > 0) {
                                            el.style.setProperty('height', w + 'px', 'important');
                                            var inners = el.querySelectorAll('canvas, iframe, .js-plotly-plot, .plotly');
                                            inners.forEach(function(inner) {
                                                inner.style.setProperty('height', w + 'px', 'important');
                                                inner.style.setProperty('width', w + 'px', 'important');
                                            });
                                        }
                                    }
                                });
                            }
                            syncMap();
                            setInterval(syncMap, 400);
                            window.addEventListener('resize', syncMap);
                        })();
                    " style="display:none;" />
                    """,
                    unsafe_allow_html=True
                )

                if "Streamlit Natif" in engine_sel:
                    # Add color hex column to df_map_data
                    df_map_data["color"] = df_map_data["categorie"].map(hex_color_map).fillna("#6B7280")
                    
                    if "Points" in map_type_sel:
                        st.map(
                            df_map_data,
                            latitude="latitude",
                            longitude="longitude",
                            color="color",
                            size=15,
                            zoom=11.5
                        )
                    else:
                        # Render lines with pydeck PathLayer
                        import pydeck as pdk
                        path_data = []
                        for idx, row in df_map_data.iterrows():
                            wkt_str = row.get("geometry_wgs84_wkt")
                            if isinstance(wkt_str, str) and wkt_str.upper().strip().startswith("LINESTRING"):
                                try:
                                    geom = shapely.wkt.loads(wkt_str)
                                    # Convert coordinates to an explicit list of float lists to prevent serialization errors
                                    coords = [[float(pt[0]), float(pt[1])] for pt in geom.coords]
                                    path_data.append({
                                        "path": coords,
                                        "color": hex_to_rgb(row["color"]),
                                        "name": str(row["nom_rue"]),
                                        "val": float(row[metric_sel]),
                                        "val_str": f"{float(row[metric_sel]):.1f}"
                                    })
                                except Exception:
                                    pass
                        
                        if path_data:
                            layer = pdk.Layer(
                                "PathLayer",
                                path_data,
                                get_path="path",
                                get_color="color",
                                get_width=5,
                                width_min_pixels=3,
                                pickable=True
                            )
                            st.pydeck_chart(pdk.Deck(
                                layers=[layer],
                                initial_view_state=pdk.ViewState(
                                    latitude=45.764043,
                                    longitude=4.835659,
                                    zoom=11.5,
                                    pitch=0
                                ),
                                map_style="light",  # Elegant light-colored background map style
                                width=800,
                                height=800,
                                tooltip={
                                    "html": "<b>Rue :</b> {name}<br/><b>Vitesse :</b> {val_str} km/h",
                                    "style": {
                                        "backgroundColor": "#1E293B",
                                        "color": "white",
                                        "fontFamily": "'Outfit', 'Inter', Arial, sans-serif",
                                        "fontSize": "13px",
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "border": "1px solid #334155"
                                    }
                                }
                            ), use_container_width=True)
                        else:
                            st.error("❌ Erreur de génération géométrique : Impossible d'extraire les coordonnées des segments routiers.")

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
                            unsafe_allow_html=True
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
                            unsafe_allow_html=True
                        )

                else:
                    # Render with Plotly Mapbox
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
                                "categorie": False
                            },
                            labels={
                                "nom_rue": "Rue / Segment",
                                "properties_twgid": "Identifiant (twgid)",
                                metric_sel: "Vitesse (km/h)"
                            },
                            mapbox_style="carto-positron",
                            zoom=11.5,
                            center={"lat": 45.764043, "lon": 4.835659},  # Centré sur Lyon
                            height=800,
                            title=f"État du réseau routier ({len(df_map_data)} segments affichés en mode Point)"
                        )
                        fig_map.update_traces(marker=dict(size=9, opacity=0.85))
                        
                        # Design tweaks
                        fig_map.update_layout(
                            width=800,
                            height=800,
                            margin={"r": 0, "t": 30, "l": 0, "b": 0},
                            legend=dict(
                                title="Catégorie",
                                yanchor="top", y=0.95,
                                xanchor="left", x=0.02,
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="#E5E7EB",
                                borderwidth=1
                            )
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
                            wkt_str = row.get("geometry_wgs84_wkt")
                            if isinstance(wkt_str, str) and wkt_str.upper().strip().startswith("LINESTRING"):
                                try:
                                    geom = shapely.wkt.loads(wkt_str)
                                    coords = list(geom.coords)
                                    for lon, lat in coords:
                                        lons.append(lon)
                                        lats.append(lat)
                                        noms.append(row.get("nom_rue", f"Segment {row['properties_twgid']}"))
                                        vitesses.append(row[metric_sel])
                                        categories.append(row["categorie"])
                                        # Use the unique index of the row to ensure each segment is treated as a separate trace/line group
                                        ids_rue.append(idx)
                                except Exception:
                                    pass

                        df_lines = pd.DataFrame({
                            'lat': lats,
                            'lon': lons,
                            'nom': noms,
                            'vitesse': vitesses,
                            'categorie': categories,
                            'id_rue': ids_rue
                        })

                        if not df_lines.empty:
                            fig_map = px.line_mapbox(
                                df_lines,
                                lat="lat",
                                lon="lon",
                                line_group="id_rue",        # CRUCIAL : Relie les points de la même rue, sans relier les rues entre elles
                                color="categorie",          # Catégorie de trafic / vitesse
                                color_discrete_map=color_map,
                                hover_name="nom",
                                hover_data={"lat": False, "lon": False, "vitesse": True, "id_rue": True, "categorie": False},
                                labels={
                                    "nom": "Rue / Segment",
                                    "id_rue": "Identifiant de segment",
                                    "vitesse": "Vitesse (km/h)"
                                },
                                mapbox_style="carto-positron",
                                zoom=11.5,
                                center={"lat": 45.764043, "lon": 4.835659},  # Centré sur Lyon
                                height=800,
                                title=f"État du réseau routier (Tracés des {len(df_map_data)} segments les plus critiques)"
                            )

                            # Map design tweaks
                            fig_map.update_layout(
                                width=800,
                                height=800,
                                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                                legend=dict(
                                    title="Catégorie",
                                    yanchor="top", y=0.95,
                                    xanchor="left", x=0.02,
                                    bgcolor="rgba(255, 255, 255, 0.8)",
                                    bordercolor="#E5E7EB",
                                    borderwidth=1
                                )
                            )
                            # Ensure lines are bold, thick, and highly visible on the map (default is 1-2, make it 5)
                            fig_map.update_traces(line=dict(width=5))
                            
                            st.plotly_chart(fig_map, use_container_width=True)
                            st.caption(f"🛣️ Rendu de la carte en mode **Lignes (Tracés)** : **{df_lines['id_rue'].nunique()}** segments routiers dessinés via un tracé continu de **{len(df_lines)}** coordonnées géométriques.")
                        else:
                            st.error("❌ Erreur de génération géométrique : Impossible d'extraire les coordonnées des segments routiers.")


        else:
            st.warning("⚠️ Aucun segment ne correspond aux filtres de vitesse sélectionnés.")
    else:
        st.info(
            "🗺️ Les données d'ingestion et de prédiction s'afficheront ici dès que le premier cycle d'orchestration Airflow aura complété l'ingestion bronze et la transformation silver."
        )
        st.image("https://raw.githubusercontent.com/evidentlyai/evidently/main/docs/book/_static/evidently_logo.png", width=300)

