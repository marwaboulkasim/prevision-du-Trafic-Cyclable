import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X, y, baseline=True):
    """
    Évalue le modèle sur X, y.
    Retourne un dict de métriques et trace les courbes.
    """

    # Prédictions modèle
    preds = model.predict(X)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    mape = np.mean(np.abs((y - preds) / (y + 1e-6))) * 100  # éviter division par zéro

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    print("Évaluation du modèle :")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # Baseline : prédiction moyenne
    if baseline:
        baseline_pred = np.full_like(y, y.mean())
        rmse_base = np.sqrt(mean_squared_error(y, baseline_pred))
        mae_base = mean_absolute_error(y, baseline_pred)
        mape_base = np.mean(np.abs((y - baseline_pred) / (y + 1e-6))) * 100
        metrics["RMSE_baseline"] = rmse_base
        metrics["MAE_baseline"] = mae_base
        metrics["MAPE_baseline"] = mape_base
        print("\nBaseline (mean) :")
        print(f"RMSE: {rmse_base:.3f}, MAE: {mae_base:.3f}, MAPE: {mape_base:.2f}%")

    # Courbes
    plt.figure(figsize=(12,5))
    plt.plot(y.values, label="Réel")
    plt.plot(preds, label="Prédit")
    if baseline:
        plt.plot(baseline_pred, label="Baseline", linestyle="--")
    plt.title("Comparaison Réel vs Prédit")
    plt.xlabel("Observations")
    plt.ylabel("Valeur cible")
    plt.legend()
    plt.show()

    return metrics
