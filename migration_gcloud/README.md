# ☁️ Guide de Migration vers Google Cloud Platform (GCP)

Ce dossier contient le code nécessaire pour héberger le pipeline d'ingestion et de transformation de données routières sur **Google Cloud Functions (Gen 2)**, en poussant les résultats dans **BigQuery** et en sauvegardant les fichiers CSV et GeoJSON sur **Google Cloud Storage (GCS)**.

## 📁 Architecture des répertoires

* [gcf_ingest/](file:///C:/Users/Admin/Desktop/JEDHA/DEMODAY/migration_gcloud/gcf_ingest) :
  * [main.py](file:///C:/Users/Admin/Desktop/JEDHA/DEMODAY/migration_gcloud/gcf_ingest/main.py) : Récupère les données WFS du Grand Lyon et les insère dans BigQuery `bronze.trafic_vitesse_brute`.
  * [requirements.txt](file:///C:/Users/Admin/Desktop/JEDHA/DEMODAY/migration_gcloud/gcf_ingest/requirements.txt) : Dépendances d'ingestion.
* [gcf_transform/](file:///C:/Users/Admin/Desktop/JEDHA/DEMODAY/migration_gcloud/gcf_transform) :
  * [main.py](file:///C:/Users/Admin/Desktop/JEDHA/DEMODAY/migration_gcloud/gcf_transform/main.py) : Lit le dernier snapshot brut depuis BigQuery, effectue l'interpolation spatiale (Lambert 93 -> WGS84), calcule l'indexation H3, catégorise la vitesse (seuil 15/30), téléverse la sauvegarde sur GCS, et écrit le résultat dans BigQuery `silver.trafic_vitesse_propre`.
  * [requirements.txt](file:///C:/Users/Admin/Desktop/JEDHA/DEMODAY/migration_gcloud/gcf_transform/requirements.txt) : Dépendances géospatiales lourdes (Geopandas, H3, Shapely, Pyproj).

---

## 🛠️ Étape 1 : Prérequis GCP

1. **Installez le SDK gcloud** sur votre machine locale et authentifiez-vous :
   ```bash
   gcloud auth login
   gcloud config set project VOTRE_PROJECT_ID
   ```
2. **Activez les API requises** dans votre projet GCP :
   ```bash
   gcloud services enable cloudfunctions.googleapis.com \
                          run.googleapis.com \
                          artifactregistry.googleapis.com \
                          bigquery.googleapis.com \
                          storage.googleapis.com \
                          cloudscheduler.googleapis.com
   ```
3. **Créez un bucket GCS** pour stocker les fichiers CSV et GeoJSON :
   ```bash
   gcloud storage buckets create gs://lyonflow-historical-data --location=europe-west9
   ```

---

## 🚀 Étape 2 : Déploiement des Cloud Functions

### 1. Déployer la fonction d'Ingestion (`gcf_ingest`)
Exécutez cette commande depuis le terminal dans le sous-dossier `gcf_ingest/` :

```bash
gcloud functions deploy lyonflow-ingest \
  --gen2 \
  --runtime=python310 \
  --region=europe-west9 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point=ingest_traffic_data_gcf \
  --set-env-vars="API_LOGIN=votre_login,API_PASSWORD=votre_mot_de_passe,BQ_PROJECT_ID=votre_project_id"
```

### 2. Déployer la fonction de Transformation (`gcf_transform`)
Exécutez cette commande depuis le terminal dans le sous-dossier `gcf_transform/` :

```bash
gcloud functions deploy lyonflow-transform \
  --gen2 \
  --runtime=python310 \
  --region=europe-west9 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point=transform_traffic_data_gcf \
  --memory=1Gi \
  --timeout=300s \
  --set-env-vars="BQ_PROJECT_ID=votre_project_id,GCS_BUCKET_NAME=lyonflow-historical-data"
```

*Note : La transformation effectue des calculs géospatiaux (H3 et polygonisation), il est donc recommandé de lui allouer au moins 1 Go de RAM et d'étendre le timeout.*

---

## ⏰ Étape 3 : Planification automatique (Cloud Scheduler)

Pour remplacer les Dags Airflow locaux et exécuter la collecte toutes les 5 minutes, configurez deux tâches de planification.

### 1. Planifier l'ingestion (Toutes les 5 minutes)
```bash
gcloud scheduler jobs create http lyonflow-ingest-trigger \
  --schedule="*/5 * * * *" \
  --uri="URL_DE_VOTRE_FUNCTION_INGEST" \
  --http-method=POST \
  --location=europe-west9
```

### 2. Planifier la transformation (Toutes les 5 minutes, décalé de 2 minutes)
Pour laisser le temps à l'ingestion de s'exécuter, on lance la transformation à la minute 2, 7, 12, etc. :
```bash
gcloud scheduler jobs create http lyonflow-transform-trigger \
  --schedule="2,7,12,17,22,27,32,37,42,47,52,57 * * * *" \
  --uri="URL_DE_VOTRE_FUNCTION_TRANSFORM" \
  --http-method=POST \
  --location=europe-west9
```

*(Remplacez `URL_DE_VOTRE_FUNCTION_...` par l'URL HTTPS retournée par la commande de déploiement gcloud).*
