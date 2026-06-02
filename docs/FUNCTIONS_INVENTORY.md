# 📋 Inventaire des fonctions du projet LyonFlow

**Total : 71 fonctions** réparties sur **22 fichiers** Python.

_Inventaire généré automatiquement par AST (Python 3.14)._
_La "1ʳᵉ ligne de docstring" est le résumé existant après ajout des commentaires en français._

---

## `app.py` (4 fonctions)

- **L74** — `get_mlflow_runs()`  
  > Récupère les 15 derniers runs MLflow de l'expérience STGCN (id `7`).

- **L99** — `get_mlflow_artifact_plot_for_run(run_id)`  
  > Télécharge l'artifact graphique d'analyse d'erreur stratifiée d'un run MLflow.

- **L132** — `get_mlflow_metrics_history_for_run(run_id)`  
  > Construit l'historique par époque des métriques d'entraînement pour un run MLflow.

- **L178** — `plot_training_curves(df_metrics)`  
  > Génère la figure Plotly des courbes d'apprentissage (perte + MAE).


## `dags/dag_pipeline.py` (8 fonctions)

- **L45** — `transform_line_to_point(ligne_2154)`  
  > Échantillonne une `LineString` Shapely en points équidistants de 7 mètres.

- **L81** — `create_merged_polygon_from_hexes(h3_id_list)`  
  > Fusionne un ensemble de cellules H3 en un polygone Shapely unique.

- **L123** — `get_speed_category(speed)`  
  > Catégorise une vitesse (km/h) en libellé lisible.

- **L146** — `ingest_traffic_data()`  
  > Tâche Airflow #1 — Ingestion temps réel depuis l'API Grand Lyon.

- **L205** — `transform_traffic_data()`  
  > Tâche Airflow #2 — Transformation Bronze → Silver.

- **L386** — `materialize_gold_layer()`  
  > Tâche Airflow #3 — Construction de la couche Gold (faits + dimensions GNN).

- **L691** — `export_to_csv_task()`  
  > Tâche Airflow #4 — Export des données Gold vers des CSV plats.

- **L712** — `trigger_stgcn_prediction_on_ray()`  
  > Tâche Airflow #5 — Soumission et surveillance d'un job d'inférence STGCN sur Ray.


## `tests/test_cache_hits.py` (2 fonctions)

- **L32** — `h3shape_merge_cached(h3_id_list)`  
  > Variante cachée de la fusion de cellules H3 (cf. autres modules).

- **L59** — `test_runs()`  
  > Mesure le hit rate du cache spatial `_segment_spatial_cache` sur les 5 plus vieux snapshots Bronze.


## `tests/test_ingest.py` (2 fonctions)

- **L24** — `test_ingest_traffic_data_success(self, mock_create_engine, mock_get)`  
  > Cas nominal : l'API Grand Lyon renvoie un GeoJSON valide, la DB accepte l'écriture.

- **L69** — `test_ingest_traffic_data_api_failure(self, mock_get)`  
  > Cas dégradé : l'API Grand Lyon lève une `RequestException`.


## `tests/test_migrate_historical.py` (3 fonctions)

- **L30** — `test_migrate_historical_success(self, mock_read_file, mock_glob, mock_create_engine)`  
  > Cas nominal : 1 GeoJSON transformé, DB mockée, ingestion Silver.

- **L77** — `custom_to_sql(df_instance, *args, **kwargs)`  
  > (pas de docstring)

- **L129** — `test_migrate_historical_empty_folder(self, mock_glob, mock_create_engine)`  
  > Cas dégradé : aucun fichier `*_transformed.json` dans le dossier.


## `tests/test_query.py` (1 fonction)

- **L15** — `h3shape_merge_cached(h3_id_list)`  
  > Variante de `h3shape_merge_cached` avec cache module-level, utilisée ici à des fins de benchmark.


## `tests/test_stgcn_model.py` (10 fonctions)

- **L28** — `setUp(self)`  
  > (pas de docstring)

- **L42** — `test_model_initialization(self)`  
  > Vérifie l'initialisation des sous-modules du modèle.

- **L59** — `test_forward_shape_consistency(self)`  
  > Vérifie la cohérence des shapes entrée/sortie du forward.

- **L89** — `test_gradient_flow_and_clipping(self)`  
  > Vérifie la backprop et le gradient clipping sur un graphe complet.

- **L139** — `test_staircase_loss_weight_application(self)`  
  > Vérifie l'implémentation de la loss pondérée « en escalier ».

- **L173** — `test_load_graph_topology_mock(self)`  
  > Mocke la DB et vérifie `load_graph_topology`.

- **L185** — `mock_read_sql(sql, con, *args, **kwargs)`  
  > (pas de docstring)

- **L200** — `test_load_traffic_series_mock(self)`  
  > Mocke la DB et vérifie `load_traffic_series` (pivot, imputation NaN, cycliques).

- **L242** — `test_build_sliding_dataset(self)`  
  > Vérifie `build_sliding_dataset` (scaler fit, loaders, shapes des batches).

- **L291** — `test_metrics_calculation(self)`  
  > Vérifie la formule de calcul de la `epoch_loss` et de la `test_mae_kmh`.


## `tests/test_super_cache.py` (2 fonctions)

- **L22** — `h3shape_merge_cached(h3_id_list)`  
  > Variante cachée de la fusion H3 — voir `profile_rebuild.py`.

- **L49** — `run_test()`  
  > Mesure les performances du **Super Cache** sur les 5 plus vieux snapshots Bronze.


## `tests/test_transform.py` (6 fonctions)

- **L30** — `test_get_speed_category(self)`  
  > Vérifie les bornes de la fonction `get_speed_category`.

- **L47** — `test_transform_line_to_point(self)`  
  > Vérifie l'interpolation tous les 7 m d'une `LineString` Lambert-93.

- **L63** — `test_create_merged_polygon_from_hexes_empty(self)`  
  > Vérifie que la fusion de H3 retourne `None` pour entrée vide ou `None`.

- **L68** — `test_create_merged_polygon_from_hexes_valid(self)`  
  > Vérifie la fusion de cellules H3 valides (rés. 13) en polygone Shapely.

- **L86** — `test_transform_traffic_data_pipeline_and_silver_push(self, mock_to_file, mock_to_csv, mock_makedirs, mock_create_engine)`  
  > Test d'intégration de la transformation Bronze→Silver.

- **L109** — `custom_to_sql(df_instance, *args, **kwargs)`  
  > (pas de docstring)


## `training/stgcn/backfill_predictions.py` (2 fonctions)

- **L52** — `init_database_table(engine)`  
  > Crée `gold.fact_predictions_traffic` et ses index si nécessaire.

- **L85** — `run_backfill()`  
  > Rattrapage historique des prédictions STGCN sur tous les pas de temps manquants.


## `training/stgcn/dataset.py` (5 fonctions)

- **L9** — `load_graph_topology(engine)`  
  > Charge la topologie du graphe routier depuis la couche Gold (PostgreSQL).

- **L48** — `load_traffic_series(engine)`  
  > Charge les séries de vitesse et calcule les features temporelles cycliques.

- **L92** — `build_sliding_dataset(vitesse_matrix_raw, hour_sin, hour_cos, day_sin, day_cos, seq_len, edge_index_tensor, num_nodes, test_split, batch_size, horizons)`  
  > Construit le dataset glissant multi-horizon au format PyTorch Geometric.

- **L171** — `load_graph_topology_from_csv(folder_path)`  
  > Variante CSV de `load_graph_topology` (mode fichier, sans PostgreSQL).

- **L207** — `load_traffic_series_from_csv(folder_path)`  
  > Variante CSV de `load_traffic_series` (mode fichier).


## `training/stgcn/get_best_params.py` (3 fonctions)

- **L24** — `get_params_from_optuna()`  
  > Récupère les meilleurs hyperparamètres depuis la base Optuna (PostgreSQL).

- **L45** — `get_params_from_mlflow()`  
  > Fallback : récupère les meilleurs hyperparamètres via l'API MLflow.

- **L93** — `main()`  
  > Point d'entrée : récupère les hyperparamètres du champion et les exporte en shell.


## `training/stgcn/hpo_stgcn.py` (4 fonctions)

- **L34** — `get_db_url()`  
  > Construit l'URL SQLAlchemy PostgreSQL à partir des variables d'environnement.

- **L43** — `get_engine()`  
  > Crée un SQLAlchemy `Engine` à partir de `get_db_url()`.

- **L52** — `objective(trial)`  
  > Fonction objectif Optuna — un trial d'optimisation bayésienne.

- **L173** — `run_hpo()`  
  > Lance l'étude d'optimisation bayésienne Optuna (HPO).


## `training/stgcn/model.py` (2 fonctions)

- **L19** — `__init__(self, in_channels, hidden_channels, out_channels)`  
  > Initialise les sous-modules du modèle ST-GRU-GNN.

- **L41** — `forward(self, x, edge_index)`  
  > Passe forward du modèle.


## `training/stgcn/predict_stgcn.py` (2 fonctions)

- **L59** — `init_database_table(engine)`  
  > Crée la table `gold.fact_predictions_traffic` et ses index si nécessaire.

- **L110** — `run_prediction()`  
  > Pipeline complet d'inférence STGCN pour un cycle de prédiction.


## `training/stgcn/train_stgcn.py` (1 fonction)

- **L46** — `train_model()`  
  > Lance l'entraînement complet du modèle STGCN (mode production).


## `utils/backfill_rounded_wkt.py` (3 fonctions)

- **L12** — `psql_insert_execute_values(table, conn, keys, data_iter)`  
  > Méthode d'insertion `to_sql` optimisée via `execute_values` (insert batch multi-lignes).

- **L37** — `round_wkt(wkt_str)`  
  > Arrondit toutes les coordonnées d'un WKT à 6 décimales (~ 11 cm à l'équateur).

- **L60** — `backfill_with_rounded_wkt()`  
  > Backfill des colonnes `properties_gid` / `properties_twgid` par jointure WKT.


## `utils/create_ref_segments.py` (1 fonction)

- **L14** — `generate_mapping_exhaustive()`  
  > Construit la table de référence `silver.ref_segments` à partir de tout l'historique Bronze.


## `utils/export_db_to_csv.py` (1 fonction)

- **L21** — `run_export()`  
  > Exporte la couche Gold de PostgreSQL vers trois fichiers CSV plats.


## `utils/migrate_historical_to_silver.py` (2 fonctions)

- **L36** — `migrate_historical()`  
  > Ré-ingeste les GeoJSON historiques (`*_transformed.json`) vers la table Silver.

- **L104** — `to_json_str(val)`  
  > Convertit n'importe quelle valeur en chaîne JSON valide.


## `utils/profile_rebuild.py` (3 fonctions)

- **L22** — `h3shape_merge_cached(h3_id_list)`  
  > Variante cachée de `create_merged_polygon_from_hexes` (DAG).

- **L58** — `get_speed_category(speed)`  
  > Catégorise une vitesse (km/h) — duplicata local de la version du DAG.

- **L80** — `run_profile()`  
  > Profile la transformation Silver sur deux snapshots consécutifs.


## `utils/rebuild_silver_from_bronze.py` (4 fonctions)

- **L38** — `psql_insert_execute_values(table, conn, keys, data_iter)`  
  > Helper d'insertion batch via `psycopg2.extras.execute_values` (identique à `utils/backfill_rounded_wkt.py`).

- **L74** — `h3shape_merge_cached(h3_id_list)`  
  > Fusionne une liste de cellules H3 en polygone, avec cache module-level.

- **L107** — `get_speed_category(speed)`  
  > Catégorise une vitesse (km/h) — voir version équivalente dans le DAG.

- **L127** — `rebuild_silver()`  
  > Reconstruit intégralement `silver.trafic_vitesse_propre` depuis tout l'historique Bronze.


