from pipeline.data_loader import load_traffic_data
from pipeline.preprocessing import preprocess, split_data
from pipeline.train import train_model
from pipeline.evaluate import evaluate_model
from pipeline.save import save_model, save_encoder, save_metrics

def main():
    print("Chargement des données...")
    df = load_traffic_data()

    print("Préprocessing...")
    X, y = preprocess(df)

    print("Split train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(X_train.info())

    # print("Entraînement du modèle...")
    # model, rmse = train_model(X_train, y_train)  # bien récupérer les deux valeurs
    # print(f"RMSE train: {rmse:.2f}")
    

    # print("Évaluation...")
    # metrics = evaluate_model(model, X_val, y_val)
    # print(metrics)

    # # sauvegarde
    # save_model(model)
    # # si tu as un encoder (OneHot) tu peux le sauver, sinon ignore
    # # save_encoder(encoder)
    # save_metrics(metrics)

if __name__ == "__main__":
    main()
