from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd

from model import get_model
from preprocess import create_features
from utils import load_data, prepare_submission
from settings import *

def train(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame):
    print(f"Training {MODEL_TYPE} model...")

    model = get_model(MODEL_TYPE, random_state = RANDOM_SEED)

    if GRID_SEARCH:
        if MODEL_TYPE == "gbc":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        else:  # lr
            param_grid = {
                'C': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

        # Cross validation splitter
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
        print(f"Best parameters found: {grid_search.best_params_}")
        model = grid_search.best_estimator_
        
        best_idx = grid_search.best_index_
        mean_best = grid_search.cv_results_['mean_test_score'][best_idx]
        std_best  = grid_search.cv_results_['std_test_score'][best_idx]
        print(f"Best params mean CV: {mean_best:.4f} ± {std_best:.4f}")
    else:
        model.fit(X_train, y_train)
        print("Training model complete")
    return model

def main(train_file_path: str, test_file_path: str) -> None:
    # Load data
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)

    # Create features
    print("Processing training data...")
    train_df = create_features(train_data)
    print("Processing test data...")
    test_df = create_features(test_data)

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
    
    if MODEL_TYPE == "lr":
        # Scaling of the features for Logistic Regression
        print("Applying StandardScaler...")

        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Train model
    model = train(X_train, y_train, X_val, y_val, X_test)

    # Get model predictions (as class labels 0/1)
    y_pred = model.predict(X_val)

    # Compute accuracy
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    # Prepare submission
    test_predictions = model.predict(X_test)
    prepare_submission(test_predictions, test_df)

if __name__ == '__main__':
    main(TRAIN_FILE_PATH, TEST_FILE_PATH)
