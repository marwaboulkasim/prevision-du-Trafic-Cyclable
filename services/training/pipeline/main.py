from pipeline.data_loader import load_data_from_supabase
from pipeline.preprocessing import preprocess, split_data
from pipeline.train import train_model
from pipeline.evaluate import evaluate_model
from pipeline.save import save_model, save_metrics

def main():
    print("Chargement des données depuis Supabase...")
    df = load_data_from_supabase()

    print("Préprocessing...")
    X, y = preprocess(df)

    print("Split train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Entraînement du modèle...")
    model, rmse = train_model(X_train, y_train)
    print(f"RMSE train: {rmse:.2f}")

    print("Évaluation...")
    metrics = evaluate_model(model, X_val, y_val)
    print(metrics)

    save_model(model)
    save_metrics(metrics)

if __name__ == "__main__":
    main()
