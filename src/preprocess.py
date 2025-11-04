from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

TABLE_TYPE = {
    "Normal": ([], ["Rock", "Steel"], ["Ghost"]),
    "Fire": (["Grass", "Ice", "Bug", "Steel"],
             ["Fire", "Water", "Rock", "Dragon"], []),
    "Water": (["Fire", "Ground", "Rock"],
              ["Water", "Grass", "Dragon"], []),
    "Electric": (["Water", "Flying"],
                 ["Electric", "Grass", "Dragon"], ["Ground"]),
    "Grass": (["Water", "Ground", "Rock"],
              ["Fire", "Grass", "Poison", "Flying", "Bug", "Dragon", "Steel"], []),
    "Ice": (["Grass", "Ground", "Flying", "Dragon"],
            ["Fire", "Water", "Ice", "Steel"], []),
    "Fighting": (["Normal", "Ice", "Rock", "Dark", "Steel"],
                 ["Poison", "Flying", "Psychic", "Bug", "Fairy"], []),
    "Poison": (["Grass", "Fairy"],
               ["Poison", "Ground", "Rock", "Ghost"], []),
    "Ground": (["Fire", "Electric", "Poison", "Rock", "Steel"],
               ["Grass", "Bug"], ["Flying"]),
    "Flying": (["Grass", "Fighting", "Bug"],
               ["Electric", "Rock", "Steel"], []),
    "Psychic": (["Fighting", "Poison"],
                ["Psychic", "Steel"], ["Dark"]),
    "Bug": (["Grass", "Psychic", "Dark"],
            ["Fire", "Fighting", "Poison", "Flying", "Ghost", "Steel", "Fairy"], []),
    "Rock": (["Fire", "Ice", "Flying", "Bug"],
             ["Fighting", "Ground", "Steel"], []),
    "Ghost": (["Psychic", "Ghost"],
              ["Dark"], ["Normal"]),
    "Dragon": (["Dragon"],
               ["Steel"], ["Fairy"]),
    "Dark": (["Psychic", "Ghost"],
             ["Fighting", "Dark", "Fairy"], []),
    "Steel": (["Ice", "Rock", "Fairy"],
              ["Fire", "Water", "Electric", "Steel"], []),
    "Fairy": (["Fighting", "Dragon", "Dark"],
              ["Fire", "Poison", "Steel"], [])
}

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

def calculate_score(p1: set[str], p2: set[str]) -> Tuple[int, int]:
    adv = 0
    res = 0
    for t1 in p1:
        t1 = t1.capitalize()
        for t2 in p2:
            t2 = t2.capitalize()

            if t2 in TABLE_TYPE[t1][0]:
                adv += 1
            elif t2 in TABLE_TYPE[t1][1]:
                adv -= 1
            elif t2 in TABLE_TYPE[t1][2]:
                adv -= 2

            if t1 in TABLE_TYPE[t2][0]:
                res -= 1
            elif t1 in TABLE_TYPE[t2][1]:
                res += 1
            elif t1 in TABLE_TYPE[t2][2]:
                res += 2
    return adv, res

def calculate_type_supremacy(data: dict) -> Tuple[int, int]:
    p1_pokemons = { }
    p2_pokemons = { }

    for turn in data["battle_timeline"]:

        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]

        if p1_name not in p1_pokemons:
            p1_pokemons[p1_name] = set()
        if p2_name not in p2_pokemons:
            p2_pokemons[p2_name] = set()

        if turn["p1_move_details"] is not None:
            p1_pokemons[p1_name].add(turn["p1_move_details"]["type"])
        if turn["p2_move_details"] is not None:
            p2_pokemons[p2_name].add(turn["p2_move_details"]["type"])

        if turn["p1_pokemon_state"]["status"] == "fnt":
            del p1_pokemons[p1_name]
        if turn["p2_pokemon_state"]["status"] == "fnt":
            del p2_pokemons[p2_name]

    p1_points = [[0, 0] for _ in p1_pokemons]
    for i, p1 in enumerate(p1_pokemons.values()):
        for p2 in p2_pokemons.values():
            adv, res = calculate_score(p1, p2)
            p1_points[i][0] += adv
            p1_points[i][1] += res
    return sum(x[0] for x in p1_points), sum(x[1] for x in p1_points)

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

        features["p1_type_adv_supremacy"], features["p1_type_res_supremacy"] = calculate_type_supremacy(battle)

        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)
