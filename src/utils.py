import json
import numpy as np
import pandas as pd

import settings

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

def load_types_record(data: list[dict]):
    pokemon_types = {}

    for battle in data:
        for pokemon in battle["p1_team_details"]:
            if pokemon["name"] not in pokemon_types:
                pokemon_types[pokemon["name"]] = [_type for _type in pokemon["types"] if _type != "notype"]

        if battle["p2_lead_details"]["name"] not in pokemon_types:
            pokemon_types[battle["p2_lead_details"]["name"]] = [_type for _type in battle["p2_lead_details"]["types"] if _type != "notype"]

    with open("data/pokemon_types.json", "w") as f:
        json.dump(pokemon_types, f, indent = 8)