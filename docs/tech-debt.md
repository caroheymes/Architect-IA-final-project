# Problèmes Connus & Roadmap

## Dette Technique Critique

### Risque d'Injection SQL dans le DAG

**Fichier** : `dags/dag_pipeline.py`, fonction `materialize_gold_layer()`

La construction de la clause `IN` utilise une f-string avec concaténation directe :

```python
query_hexes = f"""
    SELECT DISTINCT ON (properties_twgid) ...
    WHERE properties_twgid IN ({",".join(["'" + str(s) + "'" for s in active_segments])});
"""
```

Les valeurs `active_segments` viennent de la base (pas de l'utilisateur), donc le risque est limité en pratique, mais le pattern reste une mauvaise pratique. Devrait utiliser des paramètres SQLAlchemy.

### Chemin Windows Hardcodé

**Fichier** : `docker-compose.yml`, ligne 35

```yaml
- D:/donnée_histo_pour_demoday:/opt/airflow/data
```

Cassé sur macOS/Linux. Doit être remplacé par un chemin relatif ou une variable d'environnement.

## Dette Technique Modérée

### Valeurs Hardcodées dans le Dashboard

**Fichier** : `app.py`

| Ligne | Problème |
|-------|----------|
| 81 | `experiment_ids=["7"]` — ID expérience MLflow en dur |
| 213, 226 | `eb4789d2e3374056aede9faa588334c8` — Run ID de fallback en dur |
| 283-284 | `trash/ideal.png`, `trash/model_metrics.png` — Chemins de fallback dans un dossier `trash/` |

### README Pollué

**Fichier** : `README.md`, lignes 104-108

Commandes Docker de debug collées en fin de fichier, hors de tout bloc de code. À nettoyer.

### Linting Permissif

**Fichier** : `pyproject.toml`

Règles Ruff ignorées volontairement pour faciliter l'adoption initiale :
- `F401` — imports inutilisés
- `F841` — variables inutilisées
- `E712` — comparaison à False (pattern pandas)
- `B006` — argument mutable par défaut
- `B007` — variable de boucle inutilisée

À resserrer progressivement.

## Améliorations Possibles

### Pipeline de Données

- [ ] Remplacer la f-string SQL par des paramètres SQLAlchemy dans `materialize_gold_layer`
- [ ] Ajouter un mécanisme de déduplication dans Bronze (éviter les doublons si le DAG retry)
- [ ] Implémenter un watermark/checkpoint pour ne transformer que les nouvelles données Bronze
- [ ] Ajouter des data quality checks entre chaque couche (Great Expectations ou custom)
- [ ] Rendre le chemin `D:/donnée_histo_pour_demoday` configurable via variable d'environnement

### Modèle ML

- [ ] Ajouter des métriques RMSE et R² dans le tracking MLflow
- [ ] Implémenter un model registry MLflow formel (staging → production)
- [ ] Ajouter des tests d'intégration sur le pipeline complet Bronze → prédiction
- [ ] Explorer l'attention temporelle (Transformer) comme alternative au GRU
- [ ] Ajouter un monitoring de drift des données en production

### Infrastructure

- [ ] Ajouter des healthchecks aux services Airflow et Streamlit dans docker-compose
- [ ] Configurer des alertes (email/Slack) en cas d'échec du DAG
- [ ] Ajouter un Prometheus + Grafana pour le monitoring des métriques système
- [ ] Documenter la procédure de déploiement Kubernetes étape par étape
- [ ] Ajouter un script de seed pour initialiser une base de démonstration

### Code & Qualité

- [ ] Nettoyer les lignes de debug du README
- [ ] Extraire les constantes hardcodées de `app.py` vers des variables d'environnement
- [ ] Resserrer progressivement les règles ruff ignorées
- [ ] Activer `check_untyped_defs = true` dans mypy
- [ ] Ajouter des tests pour les utilitaires (`utils/`)
