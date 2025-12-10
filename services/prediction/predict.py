import pandas as pd
import pickle
from common.database.database import supabase
import datetime

def load_model():
    """Charge le modèle entraîné"""
    model_path = "models/xgb_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def fetch_daily_data():
    """Récupère les données du jour depuis Supabase"""
    today = datetime.date.today().isoformat()
    
    daily_data = supabase.table("forecast_data") \
        .select("*") \
        .eq("date", today) \
        .execute()
    return daily_data


def save_predictions_to_db(predictions, date):
    """Sauvegarde les prédictions dans la colonne 'prevision' de la table forecast_data"""
    
    success_count = 0
    error_count = 0
    
    for pred in predictions:
        counter_id = pred['counter_id']
        forecast_value = int(round(pred['prediction']))
        
        try:
            result = supabase.table("forecast_data") \
                .update({"forecast": forecast_value}) \
                .eq("counter_id", counter_id) \
                .eq("date", date) \
                .execute()
            
            if result.data:
                success_count += 1
            else:
                error_count += 1
                print(f"Aucune ligne trouvée pour counter_id={counter_id}, date={date}")
        
        except Exception as e:
            error_count += 1
            print(f"Erreur pour counter_id={counter_id}: {str(e)}")
    
    print(f" {success_count} prédictions sauvegardées, {error_count} erreurs")
    
    return success_count, error_count


def predict_traffic(model, save_to_db=True):
    """Génère la prédiction du trafic vélo pour la date choisie"""
    
    daily_data = fetch_daily_data()
    if not daily_data.data:
        print(" Aucune donnée disponible")
        return []

    df = pd.json_normalize(daily_data.data)

    # Suavegarde de counter_id avant conversion
    counter_original = df['counter_id'].copy()
    print(f"\n📋 Nombre de lignes: {len(df)}")
    print(f"📋 Compteurs: {df['counter_id'].unique()[:3]}...")
    
    # Créer la colonne 'hour' si elle n'existe pas
    if 'hour' not in df.columns:
        # Option 1: Si vous avez une colonne datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
        # Option 2: Si vous voulez prédire pour une heure spécifique (ex: midi)
        else:
            print("Pas de colonne 'hour' ou 'datetime', utilisation de hour=12 par défaut")
            df['hour'] = 12  # ou une autre valeur par défaut
    
    # Features attendues par le modèle 
    model_features =  ['counter_id', 'rolling_7d', 'rolling_28d', 'lag_7d', 'lag_28d', 'temperature', 'rain', 'hour', 'day', 'month', 'year', 'weekday', 'is_weekend']
    
    # Vérifier que toutes les colonnes existent
    missing_cols = [col for col in model_features if col not in df.columns]
    if missing_cols:
        print(f"Colonnes manquantes: {missing_cols}")
        return []
    
    # Convertir en types numériques
    for col in model_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[model_features] = df[model_features].fillna(0).astype(float)

    # Prédire
    X = df[model_features].values
    print(f" Shape de X: {X.shape} (devrait être (n, 5))")
    df["prediction"] = model.predict(X)
    df['counter_id'] = counter_original

    predictions = df[["counter_id", "prediction"]].to_dict(orient="records")
    
    # Sauvegarder les prédictions dans la base de données
    if save_to_db and predictions:
        today = datetime.date.today().isoformat()
        success, errors = save_predictions_to_db(predictions, today)
        print(f"Résultat de la sauvegarde: {success} succès, {errors} erreurs")
    
    return predictions


# Test local
if __name__ == "__main__":
    model = load_model()
    predictions = predict_traffic(model)
    print("Prédictions :", predictions)












