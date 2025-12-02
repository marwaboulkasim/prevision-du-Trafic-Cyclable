import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X, y, baseline=True, figure_dir="figures"):
    """
    Évalue le modèle et compare à la baseline.
    
    Args:
        model: modèle entraîné (XGBoost)
        X: features de validation/test
        y: target réel
        baseline: bool, calcule la baseline moyenne si True
        figure_dir: dossier pour sauvegarder les figures
    Returns:
        dict: métriques
    """

    # Créer le dossier figures s'il n'existe pas
    os.makedirs(figure_dir, exist_ok=True)

    # ------------------------
    # Prédictions du modèle
    # ------------------------
    preds = model.predict(X)

    # Filtrer y > 0 pour MAPE
    mask = y > 0
    y_nonzero = y[mask]
    preds_nonzero = preds[mask]

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    mape = np.mean(np.abs((y_nonzero - preds_nonzero) / y_nonzero)) * 100

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape)
    }

    # ------------------------
    # Baseline (moyenne)
    # ------------------------
    if baseline:
        mean_val = y.mean()
        baseline_preds = np.full_like(y, mean_val, dtype=float)
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_preds))
        baseline_mae = mean_absolute_error(y, baseline_preds)
        baseline_mape = np.mean(np.abs((y_nonzero - mean_val) / y_nonzero)) * 100

        metrics.update({
            "RMSE_baseline": float(baseline_rmse),
            "MAE_baseline": float(baseline_mae),
            "MAPE_baseline": float(baseline_mape)
        })

    # ------------------------
    # Visualisation
    # ------------------------
    plt.figure(figsize=(12,6))
    plt.plot(y.values[:500], label="True", marker="o", linestyle="-")
    plt.plot(preds[:500], label="Model", marker="x", linestyle="--")
    if baseline:
        plt.plot(baseline_preds[:500], label="Baseline", linestyle=":")
    plt.title("Comparaison valeurs réelles vs prédictions")
    plt.xlabel("Index")
    plt.ylabel("Trafic cyclable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "comparison_plot.png"))
    plt.close()

    # Histogramme des erreurs
    plt.figure(figsize=(8,5))
    errors = preds - y
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title("Distribution des erreurs du modèle")
    plt.xlabel("Erreur")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "error_distribution.png"))
    plt.close()

    print("Évaluation du modèle :")
    print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.3f}")
    if baseline:
        print(f"Baseline (mean) : RMSE: {baseline_rmse:.3f}, MAE: {baseline_mae:.3f}, MAPE: {baseline_mape:.3f}")

    return metrics
