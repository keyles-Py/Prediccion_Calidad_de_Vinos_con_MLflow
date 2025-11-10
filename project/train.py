# train.py
import argparse
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

def train_logistic_regression(C, solver, max_iter):
    with mlflow.start_run(run_name=f"LR_C{C}_iter{max_iter}_{solver}"):
        lr_model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.sklearn.log_model(lr_model, "lr_model")
        print(f"LR Run completado con F1: {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LR", choices=["LR", "NN"], help="Modelo a entrenar (LR)")
    
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="liblinear")
    parser.add_argument("--max_iter", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.model == "LR":
        train_logistic_regression(args.C, args.solver, args.max_iter)