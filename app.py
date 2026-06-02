import os

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_code_info = {"page_title": "LyonFlow - Traffic Prediction", "page_icon": "🚦", "layout": "wide"}

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
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
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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
        runs = client.search_runs(experiment_ids=["7"], order_by=["attribute.start_time DESC"], max_results=15)
        return runs
    except Exception:
        # En cas d'erreur de connexion MLflow, on retourne une liste vide pour
        # que l'UI Streamlit continue de fonctionner en mode dégradé.
        return []


def get_mlflow_artifact_plot_for_run(run_id):
    """Télécharge l'artifact graphique d'analyse d'erreur stratifiée d'un run MLflow.

    Cherche le fichier `plots/stratified_error_analysis.png` dans les artifacts
    du run et le rapatrie en local pour pouvoir l'afficher dans Streamlit.

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

    # Chemin relatif de l'artifact tel qu'il a été loggé par l'entraînement.
    plot_subpath = "plots/stratified_error_analysis.png"

    try:
        local_path = client.download_artifacts(run_id, plot_subpath)
        if os.path.exists(local_path):
            return local_path
    except Exception:
        # On capture toute exception (run inexistant, artifact manquant,
        # problème réseau…) pour ne pas crasher l'UI.
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

    for r in runs:
        run_name = r.data.tags.get("mlflow.runName", "STGCN Run")
        status = r.info.status
        label = f"{run_name} ({r.info.run_id[:8]}) [{status}]"
        run_options.append(label)
        run_id_map[label] = r.info.run_id
        run_name_map[label] = run_name
        run_status_map[label] = status

    default_index = 0
    for idx, label in enumerate(run_options):
        if run_id_map[label] == "eb4789d2e3374056aede9faa588334c8":
            default_index = idx
            break

    selected_label = st.sidebar.selectbox("Sélectionner un entraînement :", run_options, index=default_index)
    selected_run_id = run_id_map[selected_label]
    selected_run_name = run_name_map[selected_label]
    selected_run_status = run_status_map[selected_label]

    status_emoji = "🟢" if selected_run_status == "FINISHED" else "🟠" if selected_run_status == "RUNNING" else "🔴"
    st.sidebar.success(f"{status_emoji} Run actif: `{selected_run_id[:8]}`")
else:
    st.sidebar.warning("⚠️ Connexion directe MLflow indisponible.")
    selected_run_id = "eb4789d2e3374056aede9faa588334c8"
    selected_run_name = "STGCN Production Run"
    selected_run_status = "FINISHED"

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
        """
        <div class="metric-card">
            <h3>Modèle de Prédiction</h3>
            <h2 style="color: #059669;">ST-GRU-GNN</h2>
            <p>Spatiotemporel (MLflow)</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <h3>Volume Ingestion</h3>
            <h2 style="color: #D97706;">En cours...</h2>
            <p>Couche Bronze lancée</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.write("---")

# Reordered Tabs: Learning curves/performances are in Tab 1 (Default Active Tab)
tab1, tab2, tab3 = st.tabs(
    ["📈 Courbes d'Apprentissage (Perte & MAE)", "🎯 Analyse d'Erreur Stratifiée", "🗺️ Visualisation Temps Réel"]
)

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
    st.subheader("🗺️ État du réseau routier en temps réel")
    st.info(
        "Les données d'ingestion et de prédiction s'afficheront ici dès que le premier cycle d'orchestration Airflow aura complété l'ingestion bronze et la transformation silver."
    )
