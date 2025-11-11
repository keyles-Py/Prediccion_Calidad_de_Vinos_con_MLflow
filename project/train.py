# train.py
import argparse
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import argparse

def load_and_prepare_data():
    df = pd.read_csv("../data/winequality-white.csv", sep=';')
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main(n_estimators, max_depth, random_state):
    run_name = f"RF_est{n_estimators}_depth{max_depth}"
    with mlflow.start_run(run_name=run_name):

        X_train, X_test, y_train, y_test = load_and_prepare_data()
        
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        mlflow.sklearn.log_model(rf_model, "rf_model")
        
        print(f"RF Run completado con RMSE: {rmse:.4f} y R2: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=100
    )
    parser.add_argument(
        "--max_depth", 
        type=int, 
        default=10
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42
    )
    
    args = parser.parse_args()
    main(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)