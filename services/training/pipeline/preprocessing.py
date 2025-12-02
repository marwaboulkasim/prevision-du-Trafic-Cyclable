from pipeline.data_loader import load_traffic_data
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TARGET_COLUMN, FEATURE_COLUMNS, DATE_COLUMN, CATEGORICAL_COLUMNS, DATE_FEATURES, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE

def extract_date_features(df, date_column=DATE_COLUMN):
    df[date_column] = pd.to_datetime(df[date_column])
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["hour"] = df[date_column].dt.hour
    jours_fr = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
    # Convertir directement en codes numériques au lieu de category
    df["weekday"] = df[date_column].dt.weekday
    df["weekday_name"] = df[date_column].dt.weekday.map(jours_fr)  # Optionnel: garder les noms pour référence
    
    return df

def prepare_features_for_xgboost(X):
    """Prépare les features pour être compatibles avec XGBoost"""
    X = X.copy()
    
    # Supprimer les colonnes ID si présentes
    id_cols = [col for col in X.columns if 'id' in col.lower()]
    if id_cols:
        X = X.drop(columns=id_cols)
        print(f"Colonnes ID supprimées: {id_cols}")
    
    # Convertir les catégories en codes numériques
    categorical_cols = X.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        X[col] = X[col].cat.codes
        print(f"Colonne '{col}' convertie de category à int")
    
    # Convertir les objets en numériques
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            X[col] = pd.to_numeric(X[col])
            print(f"Colonne '{col}' convertie de object à numeric")
        except:
            # Si la conversion échoue, utiliser factorize
            X[col] = pd.factorize(X[col])[0]
            print(f"Colonne '{col}' encodée avec factorize")
    
    return X

def preprocess(df):
    df = extract_date_features(df)

    # Sélection des features
    features = FEATURE_COLUMNS + [col for col in DATE_FEATURES if col not in FEATURE_COLUMNS]
    
    # Supprimer 'id' et 'weekday_name' des features si présents
    features = [f for f in features if f not in ['id', 'weekday_name']]
    
    X = df[features].copy()
    y = df[TARGET_COLUMN].copy()

    # Conversion des colonnes catégorielles en codes numériques
    existing_cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    for col in existing_cat_cols:
        X[col] = X[col].astype("category")
        if X[col].isna().any():
            X[col] = X[col].cat.add_categories("nan").fillna("nan")
        # Convertir immédiatement en codes numériques
        X[col] = X[col].cat.codes

    # Gérer weekday s'il est présent
    if "weekday" in X.columns:
        # weekday est déjà numérique (0-6) depuis extract_date_features
        X["weekday"] = X["weekday"].astype("int64")

    # Remplacement des NaN pour les colonnes numériques
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    # Vérification finale des types
    print("\n=== Types de données après preprocessing ===")
    print(X.dtypes)
    print(f"\nShape: {X.shape}")

    return X, y

def split_data(X, y, test_size=TEST_SIZE, val_size=VALIDATION_SIZE, random_seed=RANDOM_SEED):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_seed
    )
    
    # Appliquer la préparation finale pour XGBoost sur chaque split
    print("\n=== Préparation des données pour XGBoost ===")
    X_train = prepare_features_for_xgboost(X_train)
    X_val = prepare_features_for_xgboost(X_val)
    X_test = prepare_features_for_xgboost(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
