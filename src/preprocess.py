from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

def count_fnt(data: dict) -> Tuple[int, int]:
    out1 = 0
    out2 = 0
    for turn in data["battle_timeline"]:
        if turn["p1_pokemon_state"]["status"] == "fnt":
            out1 += 1
        if turn["p2_pokemon_state"]["status"] == "fnt":
            out2 += 1
    return out1, out2
def calculate_mean_hp(data: dict):
    p1_pokemons = { }
    p2_pokemons = { }

    for turn in data["battle_timeline"]:
        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]

        p1_pokemons[p1_name] = turn["p1_pokemon_state"]["hp_pct"]
        p2_pokemons[p2_name] = turn["p2_pokemon_state"]["hp_pct"]

    p1_mean = np.mean(list(p1_pokemons.values()))
    p2_mean = np.mean(list(p2_pokemons.values()))

    return p1_mean, p2_mean

def create_features(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    for battle in tqdm(data, desc = "Extracting features"):
        features = { }

        # --- Player 1 Team Features ---
        p1_team = battle.get('p1_team_details', [])
        if p1_team:
            features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
            features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
            features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
            features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
            features["p1_mean_spa"] = np.mean([p.get("base_spa", 0) for p in p1_team])
            features["p1_mean_spd"] = np.mean([p.get("base_spd", 0) for p in p1_team])

        p1_fnt, p2_fnt = count_fnt(battle)
        features['p1_fnt'] = p1_fnt
        features['p2_fnt'] = p2_fnt
        features["p1_p2_diff_fnt"] = p1_fnt - p2_fnt

        p1_hps, p2_hps = calculate_mean_hp(battle)
        features['p1_mean_hp_pct'] = p1_hps
        features['p2_mean_hp_pct'] = p2_hps

        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)
