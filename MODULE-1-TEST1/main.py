import os
import time
from os.path import join as join

import joblib
import mlflow  # <-- MLflow Tracing + Tracking
import pandas as pd
from mlflow import (
    log_artifact,
    log_metric,
    log_metrics,
    log_params,
    set_tag,
    update_current_trace,
)
from models.models import (
    create_nn_model,
    model_predict,
    train_model_avec_earlystopping,
)
from modules.evaluate import evaluate_performance
from modules.preprocess import preprocessing, split
from modules.print_draw import draw_loss, print_data

"""
mlflow.start_run — Suivi d expériences (Experiment Tracking)
    Niveau macro
    Un objet Run (dans l UI MLflow) avec un run_id, des params, metrics, artifacts, tags
    But: enregistrer des résultats reproductibles de vos expériences (pour comparaison, historisation).

mlflow.start_span — Tracing (profilage de flux et latences)
    Niveau micro
    Un Span dans le graphe de traces (chronologie), souvent imbriqué (nested) sous un span parent
    But: diagnostiquer les performances (latences, erreurs par étape), observer le flux d’un pipeline et corréler avec un run si besoin.

"""


# --- Configuration MLflow (UI locale sur 127.0.0.1:5000) ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nn-training-tracing")  # Traces + Runs regroupés

# --- Wrappers "tracés" sur tes fonctions existantes (sans toucher aux modules) ---
preprocessing_tr = mlflow.trace(name="preprocessing", span_type="preprocess")(
    preprocessing
)
split_tr = mlflow.trace(name="split", span_type="split")(split)
create_nn_tr = mlflow.trace(name="create_nn_model", span_type="modeling")(
    create_nn_model
)
train_es_tr = mlflow.trace(name="train_with_earlystopping", span_type="train")(
    train_model_avec_earlystopping
)
predict_tr = mlflow.trace(name="predict", span_type="inference")(model_predict)
eval_perf_tr = mlflow.trace(name="evaluate", span_type="eval")(evaluate_performance)
draw_loss_tr = mlflow.trace(name="draw_loss", span_type="viz")(draw_loss)
print_data_tr = mlflow.trace(name="print_data", span_type="report")(print_data)


@mlflow.trace(name="pipeline", span_type="pipeline", attributes={"owner": "bruno"})
def main():
    # Si un run MLflow est actif, on rattache la trace au run et on ajoute du contexte
    if mlflow.active_run():
        update_current_trace(
            tags={"stage": "dev", "component": "nn-train"},
            # metadata est immuable : parfait pour des IDs stables
            metadata={"mlflow.run_id": mlflow.active_run().info.run_id},
        )

    # Chargement des datasets
    df_old = pd.read_csv(join("data", "df_concat.csv"))
    # On logge un échantillon & quelques infos de données
    try:
        sample_path = "data_sample_head100.csv"
        df_old.head(100).to_csv(sample_path, index=False)
        log_artifact(sample_path, artifact_path="data")
        log_params(
            {"rows_total": int(df_old.shape[0]), "cols_total": int(df_old.shape[1])}
        )
    except Exception:
        pass

    # Charger le préprocesseur
    with mlflow.start_span(name="load_preprocessor", attributes={"stage": "io"}) as s:
        preprocessor_loaded = joblib.load(join("models", "preprocessor.pkl"))
        # Éviter de logguer l'objet brut (trop volumineux / pas sérialisable) :
        s.set_outputs({"preprocessor": "loaded"})

    # Prétraitement
    X, y, _ = preprocessing_tr(df_old)

    # Split
    X_train, X_test, y_train, y_test = split_tr(X, y)
    # Hyperparams de base
    log_params(
        {"early_stopping": True, "max_epochs": 3500, "input_dim": int(X_train.shape[1])}
    )

    # Créer un modèle
    model = create_nn_tr(X_train.shape[1])

    # Entraînement initial
    with mlflow.start_span(name="train_initial", attributes={"stage": "train"}) as s:
        model, hist = train_es_tr(
            model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=3500, verbose=1
        )
        # Exemple d’output résumé du span
        try:
            epochs_done = len(getattr(hist, "history", {}).get("loss", []))
        except Exception:
            epochs_done = None
        s.set_outputs({"epochs_done": epochs_done})

    # Log de l'historique d'entraînement (métriques par epoch si disponibles)
    history = getattr(hist, "history", None)
    if isinstance(history, dict):
        if "loss" in history:
            for i, v in enumerate(history["loss"]):
                log_metric("loss", float(v), step=i)
        if "val_loss" in history:
            for i, v in enumerate(history["val_loss"]):
                log_metric("val_loss", float(v), step=i)

    # Visualisation des pertes
    draw_loss_tr(hist)

    # Si la fonction a écrit un PNG courant dans le dossier, on le loggue (best-effort)
    for p in ["loss.png", "loss_curve.png", "training_curves.png"]:
        if os.path.exists(p):
            log_artifact(p, artifact_path="figures")
            break

    # Sauvegarder le modèle
    joblib.dump(model, join("models", "model_2026_newmodel.pkl"))

    # Log du modèle comme artefact (framework agnostique)
    try:
        log_artifact(join("models", "model_2026_newmodel.pkl"), artifact_path="models")
    except Exception:
        pass

    # Recharger le modèle
    model_2026 = joblib.load(join("models", "model_2026_newmodel.pkl"))

    # Prédictions train + évaluation
    y_pred = predict_tr(model_2026, X_train)
    perf = eval_perf_tr(y_train, y_pred)
    print_data_tr(perf)

    # Log métriques train
    try:
        log_metrics(
            {
                f"train_{k}": float(v)
                for k, v in perf.items()
                if isinstance(v, (int, float))
            }
        )
    except Exception:
        pass

    # Prédictions test + évaluation
    y_pred = predict_tr(model_2026, X_test)
    perf = eval_perf_tr(y_test, y_pred)
    print_data_tr(perf)

    # Log métriques test
    try:
        log_metrics(
            {
                f"test_{k}": float(v)
                for k, v in perf.items()
                if isinstance(v, (int, float))
            }
        )
    except Exception:
        pass

    # WARNING ZONE : ré-entraîner sur les mêmes données (pour comparaison)
    with mlflow.start_span(
        name="warning_zone_retrain", attributes={"stage": "train"}
    ) as s:
        model2, hist2 = train_es_tr(
            model_2026, X_train, y_train, X_val=X_test, y_val=y_test
        )
        y_pred2 = predict_tr(model_2026, X_test)
        perf2 = eval_perf_tr(y_test, y_pred2)
        print_data_tr(perf2, exp_name="exp 2")
        draw_loss_tr(hist2)

        # Log métriques & figure de la 2e expérience
        try:
            log_metrics(
                {
                    f"test_v2_{k}": float(v)
                    for k, v in perf2.items()
                    if isinstance(v, (int, float))
                }
            )
        except Exception:
            pass


if __name__ == "__main__":
    # --- RUN MLflow : regroupe tous les logs de cette exécution ---
    run_name = f"nn-train-{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        set_tag("owner", "bruno")
        set_tag("code_path", os.path.abspath(__file__))
        # Exécution du pipeline (les spans/traces seront associés au run)
        main()
        # Récupérer l'ID de la dernière trace pour faciliter la navigation Run <-> Trace
        trace_id = mlflow.get_last_active_trace_id()
        set_tag("trace_id", trace_id)
        print("Dernière trace MLflow:", trace_id)
