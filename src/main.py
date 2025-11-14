import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from model import get_model
from preprocess2 import create_features
from settings import *

def load_data(path: str):
    data = []
    # Read the file line by line
    print(f"Loading data from '{path}'...")
    try:
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Successfully loaded {len(data)} battles.")
    except FileNotFoundError:
        print(f"ERROR: Could not find the file at '{data}'.")
        print("Please make sure you have added the competition data to this notebook.")
    return data

def prepare_submission(predictions: np.ndarray, test_df: pd.DataFrame) -> None:
    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': predictions
    })

    # Save the DataFrame to a .csv file
    submission_df.to_csv('submission.csv', index = False)

    print("\n'submission.csv' file created successfully!")

def main():
    train_file_path = "data/train.jsonl"
    test_file_path = "data/test.jsonl"

    # Load data
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)

    # Create features
    print("Processing training data...")
    train_df = create_features(train_data)
    print("Processing test data...")
    test_df = create_features(test_data)

    # Solo per notebook
    # print("\nTraining features preview:")
    # display(train_df.head())

    # Split into train and validation sets
    features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
    X_train = train_df[features]
    y_train = train_df['player_won']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size = 1 - TRAIN_SPLIT,
        random_state = RANDOM_SEED,
        shuffle = True
    )

    X_test = test_df[features]

    # Train model

    print(f"Training {MODEL_TYPE} model...")

    model = get_model(MODEL_TYPE, random_state = RANDOM_SEED)

    ## GRID SEARCH
    if GRID_SEARCH:  # per gbc è lenta
        if MODEL_TYPE == "gbc":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]  # Frazione di campioni per ogni albero
            }
        elif MODEL_TYPE == "xgb":
            param_grid = {
                'n_estimators': [150, 200, 250],  # Numero di alberi
                'max_depth': [3, 5],  # Profondità massima
                'learning_rate': [0.05, 0.1],  # Tasso di apprendimento
                'subsample': [0.8, 1.0],  # Frazione di campioni usati per albero
                'colsample_bytree': [0.8, 1.0]  # Frazione di feature usate per albero
            }
        else:  # lr
            # --- 1. SCALING DEI DATI (Necessario per LR) ---
            print("Applicazione dello StandardScaler...")
            # Crea lo scaler
            scaler = StandardScaler()
            # Adattalo SOLO su X_train
            scaler.fit(X_train)
            # Trasforma sia X_train che X_val
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            param_grid = {
                'C': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

        cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

        grid_search = GridSearchCV(
            estimator = model,
            param_grid = param_grid,
            cv = cv_splitter,
            scoring = 'accuracy',
            n_jobs = -1,
            verbose = 0
        )

        grid_search.fit(X_train, y_train)

        print("Training model complete")
        print(f"Migliori parametri trovati: {grid_search.best_params_}")
        # print(f"Miglior score (accuracy) dalla cross-validation: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
        
        best_idx = grid_search.best_index_
        mean_best = grid_search.cv_results_['mean_test_score'][best_idx]
        std_best  = grid_search.cv_results_['std_test_score'][best_idx]
        print(f"Best params mean CV: {mean_best:.4f} ± {std_best:.4f}")
    else:
        model.fit(X_train, y_train)
        print("Training model complete")

    # Get model predictions (as class labels 0/1)
    y_pred = model.predict(X_val)

    # Compute accuracy
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Prepare submission
    test_predictions = model.predict(X_test)
    prepare_submission(test_predictions, test_df)

if __name__ == '__main__':
    main()
