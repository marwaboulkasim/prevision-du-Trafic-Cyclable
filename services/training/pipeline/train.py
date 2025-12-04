import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(X_train, y_train):
    # convertir object -> category
    for col in X_train.select_dtypes(include="object").columns:
        X_train[col] = X_train[col].astype("category")
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        enable_categorical=True  
    )

    model.fit(X_train, y_train, verbose=False)
    
    # calculer RMSE sur train pour info
    y_pred = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    
    return model, rmse
