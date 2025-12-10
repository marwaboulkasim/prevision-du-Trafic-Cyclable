import pandas as pd
from sklearn.model_selection import train_test_split
from .config import (
    TARGET_COLUMN, FEATURE_COLUMNS, DATE_COLUMN,
    CATEGORICAL_COLUMNS, DATE_FEATURES,
    RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE
)

# Extraction des features de date
def extract_date_features(df, date_column=DATE_COLUMN):
    if date_column not in df.columns:
        raise KeyError(f"Colonne {date_column} introuvable dans le dataframe")
    
    df[date_column] = pd.to_datetime(df[date_column])
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["hour"] = df[date_column].dt.hour
    df["weekday"] = df[date_column].dt.weekday
    return df

# Préparation pour XGBoost
def prepare_features_for_xgboost(X, keep_ids=True):
    X = X.copy()

    if not keep_ids:
        # Supprimer les colonnes ID si on ne veut pas les garder
        id_cols = [col for col in X.columns if 'id' in col.lower()]
        if id_cols:
            X = X.drop(columns=id_cols)
            print(f"Colonnes ID supprimées: {id_cols}")
    else:
        print("Colonnes ID conservées pour le modèle.")

    # Convertir les catégories en codes numériques
    categorical_cols = X.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        X[col] = X[col].cat.codes
        print(f"Colonne '{col}' convertie de category à int")

    # Convertir les objets en numériques si possible, sinon factorize
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            X[col] = pd.to_numeric(X[col])
            print(f"Colonne '{col}' convertie de object à numeric")
        except:
            X[col] = pd.factorize(X[col])[0]
            print(f"Colonne '{col}' encodée avec factorize")

    return X

# Préprocessing complet
def preprocess(df, keep_ids=True):
    print("Préprocessing des données...")
    df = extract_date_features(df)

    # Sélection des features
    features = FEATURE_COLUMNS + [col for col in DATE_FEATURES if col not in FEATURE_COLUMNS]
    features = [f for f in features if f in df.columns]  # uniquement celles existantes

    X = df[features].copy()
    y = df[TARGET_COLUMN].copy()

    # Conversion des colonnes catégorielles définies dans config
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Remplissage des NaN pour colonnes numériques
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    # Vérification finale
    print("\n=== Colonnes utilisées pour le modèle ===")
    print(X.columns.tolist())
    print("\n=== Types de données après preprocessing ===")
    print(X.dtypes)
    print(f"\nShape: {X.shape}")

    # Préparation finale pour XGBoost
    X = prepare_features_for_xgboost(X, keep_ids=keep_ids)
    return X, y

# Split train/val/test
def split_data(X, y, test_size=TEST_SIZE, val_size=VALIDATION_SIZE, random_seed=RANDOM_SEED):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_seed
    )

    print("\n=== Split train/val/test réalisé ===")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test
