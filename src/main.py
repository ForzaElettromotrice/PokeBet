import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import get_model
from preprocess import create_features

TRAIN_SPLIT = 0.9
RANDOM_SEED = 42

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
    print("Training a model...")
    model = get_model("gbc", random_state = RANDOM_SEED)
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
