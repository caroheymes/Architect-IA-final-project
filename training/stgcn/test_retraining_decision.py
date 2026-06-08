"""
test_retraining_decision.py
===========================
Script autonome permettant de tester la logique d'évaluation de la dérive (drift), 
l'extraction des hyperparamètres optimaux, et le déclenchement d'un réentraînement 
sécurisé (dry-run ou test rapide à 1 époque) sans risque d'erreur.

USAGE :
    # Simulation simple (dry-run par défaut) :
    python training/stgcn/test_retraining_decision.py

    # Forcer le réentraînement même si les métriques sont bonnes :
    python training/stgcn/test_retraining_decision.py --force

    # Lancer un réentraînement réel de test (sécurisé, limité à 1 époque) :
    python training/stgcn/test_retraining_decision.py --force --epochs 1 --no-dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import sys

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamFormatter(sys.stdout) if hasattr(logging, "StreamFormatter") else logging.StreamHandler()]
)
logger = logging.getLogger("LyonFlow-Test-Retraining")

# Ajout du répertoire courant au PYTHONPATH pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import get_best_params
except ImportError:
    get_best_params = None
    logger.warning("⚠️ Impossible d'importer 'get_best_params.py' directement. Assurez-vous d'exécuter le script depuis la racine du projet.")

# Valeurs par défaut des chemins
DEFAULT_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "out", "monitoring_metrics_morning.json"
)


def load_metrics(metrics_path):
    """Charge le fichier JSON contenant les métriques de monitoring d'Evidently AI."""
    if not os.path.exists(metrics_path):
        logger.error(f"❌ Le fichier de métriques est introuvable : {metrics_path}")
        return None
    
    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture du fichier JSON : {e}")
        return None


def extract_key_metrics(metrics_data):
    """Extrait la MAE moyenne et la p-value de dérive (Kolmogorov-Smirnov)."""
    mae_value = None
    p_value_drift = None
    
    metrics_list = metrics_data.get("metrics", [])
    for m in metrics_list:
        metric_name = m.get("metric_name", "")
        # Extraction de la MAE
        if "MAE(regression_name" in metric_name:
            val = m.get("value", {})
            if isinstance(val, dict):
                mae_value = val.get("mean")
            else:
                mae_value = val
                
        # Extraction de la dérive (ValueDrift)
        elif "ValueDrift" in metric_name or m.get("config", {}).get("type") == "evidently:metric_v2:ValueDrift":
            p_value_drift = m.get("value")
            
    return mae_value, p_value_drift


def get_optimal_hyperparameters():
    """Récupère les hyperparamètres recommandés depuis Optuna ou MLflow avec fallbacks."""
    params = {}
    if get_best_params is not None:
        # Tente de charger depuis Optuna
        optuna_params = get_best_params.get_params_from_optuna()
        if optuna_params:
            params = optuna_params
            logger.info("🎯 Hyperparamètres chargés avec succès depuis la base de données Optuna.")
        else:
            # Fallback sur MLflow
            mlflow_params = get_best_params.get_params_from_mlflow()
            if mlflow_params:
                params = mlflow_params
                logger.info("🎯 Hyperparamètres chargés avec succès depuis MLflow.")
                
    if not params:
        logger.warning("⚠️ Aucun paramètre trouvé en base/MLflow. Application des valeurs par défaut sécurisées.")
        params = {
            "learning_rate": 0.001,
            "hidden_channels": 128,
            "weight_decay": 1e-5,
            "batch_size": 2,
            "seq_len": 120,
            "weight_jam": 15.0,
            "weight_slow": 5.0
        }
    
    # Post-traitements de sécurité identiques à get_best_params.py
    params["seq_len"] = 120
    if "batch_size" in params and params["batch_size"] > 16:
        params["batch_size"] = 16
        
    return params


def main():
    parser = argparse.ArgumentParser(description="Test autonome et sécurisé de la décision de réentraînement.")
    parser.add_argument("--metrics-path", type=str, default=DEFAULT_METRICS_PATH,
                        help="Chemin vers le fichier de métriques Evidently JSON.")
    parser.add_argument("--force", action="store_true",
                        help="Force le déclenchement du réentraînement sans vérifier les seuils.")
    parser.add_argument("--dry-run", action="store_true", default=True, dest="dry_run",
                        help="Mode Dry-Run (par défaut) : affiche uniquement les actions sans exécuter d'entraînement.")
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run",
                        help="Désactive le mode Dry-Run et lance l'entraînement de test réel.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Nombre d'époques pour l'entraînement de test (recommandé: 1 pour valider sans lenteur).")
    parser.add_argument("--mae-threshold", type=float, default=5.0,
                        help="Seuil maximal de MAE toléré avant réentraînement (défaut: 5.0 km/h).")
    parser.add_argument("--p-value-threshold", type=float, default=0.05,
                        help="Seuil de p-value en dessous duquel on considère qu'il y a dérive (défaut: 0.05).")
    
    args = parser.parse_args()
    
    print("\n" + "═"*70)
    print(" 🛡️  SIMULATEUR DE DÉCISION & RÉENTRAÎNEMENT DE SÉCURITÉ — LYONFLOW")
    print("═"*70)
    
    # 1. Chargement et Analyse des Métriques
    logger.info(f"Chargement des métriques depuis : {args.metrics_path}")
    metrics_data = load_metrics(args.metrics_path)
    
    mae = None
    p_value = None
    trigger_needed = False
    reasons = []
    
    if metrics_data:
        mae, p_value = extract_key_metrics(metrics_data)
        
        # Affichage des métriques lues
        print("\n📊 ÉTAT DES MÉTRIQUES DE MONITORING :")
        if mae is not None:
            print(f"  • MAE moyenne du matin : {mae:.4f} km/h (Seuil max : {args.mae-threshold} km/h)")
            if mae > args.mae_threshold:
                trigger_needed = True
                reasons.append(f"Précision dégradée (MAE de {mae:.2f} > {args.mae_threshold:.2f} km/h)")
        else:
            print("  • MAE moyenne : Introuvable dans le JSON")
            
        if p_value is not None:
            print(f"  • p-value Dérive de données : {p_value:.6f} (Seuil critique : < {args.p-value-threshold})")
            if p_value < args.p_value_threshold:
                trigger_needed = True
                reasons.append(f"Dérive de données statistiquement significative (p-value {p_value:.6f} < {args.p-value-threshold})")
        else:
            print("  • p-value Dérive : Introuvable dans le JSON")
    else:
        logger.warning("⚠️ Données de métriques indisponibles. Impossible d'évaluer automatiquement l'état.")
        
    if args.force:
        trigger_needed = True
        reasons.append("Déclenchement forcé manuellement via l'option --force")
        
    print("\n" + "─"*70)
    
    # 2. Prise de Décision
    if trigger_needed:
        print("🚨 RÉSULTAT : UN RÉENTRAÎNEMENT DU MODÈLE EST REQUIS")
        print("Raisons du déclenchement :")
        for reason in reasons:
            print(f"  ➡️  [DECLENCHEUR] {reason}")
    else:
        print("✅ RÉSULTAT : LE MODÈLE RESTE PERFORMANT. AUCUN RÉENTRAÎNEMENT REQUIS.")
        print("  ➡️  [INFO] Les métriques sont au-dessus des seuils de dérive et d'erreur.")
        print("  💡  (Utilisez '--force' si vous voulez tester le pipeline malgré tout !)")
        return
        
    print("─"*70 + "\n")
    
    # 3. Récupération des Hyperparamètres Optimaux
    logger.info("Recherche des hyperparamètres HPO optimaux...")
    best_params = get_optimal_hyperparameters()
    
    print("🎯 PARAMÈTRES RETENUS POUR L'ENTRAÎNEMENT :")
    for k, v in best_params.items():
        print(f"  • {k.upper()} = {v}")
        
    print("\n" + "─"*70)
    
    # Préparation des variables d'environnement
    env = os.environ.copy()
    env["USE_LOCAL_CSV"] = "true"
    env["DATA_FOLDER"] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/in"))
    env["EPOCHS"] = str(args.epochs)
    env["SEQ_LEN"] = str(best_params.get("seq_len", 120))
    env["BATCH_SIZE"] = str(best_params.get("batch_size", 2))
    env["HIDDEN_CHANNELS"] = str(best_params.get("hidden_channels", 128))
    env["DROPOUT"] = str(best_params.get("dropout", 0.1))
    
    # On gère l'écriture de lr en LEARNING_RATE
    lr_val = best_params.get("learning_rate") or best_params.get("lr") or 0.001
    env["LEARNING_RATE"] = str(lr_val)
    
    if "weight_decay" in best_params:
        env["WEIGHT_DECAY"] = str(best_params["weight_decay"])
    if "weight_jam" in best_params:
        env["WEIGHT_JAM"] = str(best_params["weight_jam"])
    if "weight_slow" in best_params:
        env["WEIGHT_SLOW"] = str(best_params["weight_slow"])
        
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_stgcn_v2.py")
    
    # Commande représentative
    env_vars_str = " ".join([f"{k}={v}" for k, v in env.items() if k in [
        "USE_LOCAL_CSV", "DATA_FOLDER", "EPOCHS", "SEQ_LEN", "BATCH_SIZE", 
        "HIDDEN_CHANNELS", "LEARNING_RATE", "WEIGHT_DECAY"
    ]])
    representative_cmd = f"{env_vars_str} python {os.path.basename(train_script)}"
    
    # 4. Exécution (Mode Dry-Run ou Exécution Réelle)
    if args.dry_run:
        print("💡 [MODE DRY-RUN ACTIF] L'entraînement n'a pas été lancé.")
        print("Pour exécuter réellement ce test sécurisé à 1 époque, lancez la commande suivante :")
        print(f"👉 python training/stgcn/test_retraining_decision.py --force --epochs 1 --no-dry-run")
        print("\nCommande bash équivalente simulée :")
        print(f"👉 {representative_cmd}")
    else:
        print(f"🚀 [EXÉCUTION] Lancement de l'entraînement de test réel (Limité à {args.epochs} époque(s))")
        logger.info(f"Script exécuté : {train_script}")
        
        # S'assurer que le dossier des modèles existe
        models_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models"))
        os.makedirs(models_dir, exist_ok=True)
        
        # Forcer des chemins de sortie de test pour éviter d'écraser la production accidentellement
        env["MODEL_OUT"] = os.path.join(models_dir, "stgcn_v2_test_retraining.pt")
        env["SCALER_OUT"] = os.path.join(models_dir, "stgcn_v2_test_scaler.pkl")
        
        try:
            logger.info("Démarrage du processus d'entraînement...")
            # On exécute train_stgcn_v2.py avec l'interpréteur courant
            result = subprocess.run(
                [sys.executable, train_script],
                env=env,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            print("\n" + "═"*70)
            print("🎉 LE TEST D'ENTRAÎNEMENT S'EST TERMINÉ AVEC SUCCÈS !")
            print("  • Le script d'entraînement v2 a validé tout le pipeline sans aucune erreur de runtime.")
            print(f"  • Poids de test sauvegardés sous : {env['MODEL_OUT']}")
            print(f"  • Scaler de test sauvegardé sous : {env['SCALER_OUT']}")
            print("═"*70)
        except subprocess.CalledProcessError as e:
            print("\n" + "❌"*35)
            logger.error(f"Erreur durant l'exécution du script d'entraînement ! Code retour : {e.returncode}")
            print("❌"*35)
            sys.exit(e.returncode)
        except Exception as e:
            logger.error(f"Erreur inattendue : {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
