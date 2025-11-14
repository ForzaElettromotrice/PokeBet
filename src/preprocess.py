import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import settings

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
POKEMON_TYPES = { }

def load_types_record(data: list[dict]):
    global POKEMON_TYPES

    if not settings.CREATE_TYPES_RECORD:
        with open("data/pokemon_types.json") as f:
            POKEMON_TYPES = json.load(f)
            return

    for battle in data:
        for pokemon in battle["p1_team_details"]:
            if pokemon["name"] not in POKEMON_TYPES:
                POKEMON_TYPES[pokemon["name"]] = [_type for _type in pokemon["types"] if _type != "notype"]

        if battle["p2_lead_details"]["name"] not in POKEMON_TYPES:
            POKEMON_TYPES[battle["p2_lead_details"]["name"]] = [_type for _type in battle["p2_lead_details"]["types"] if _type != "notype"]

    with open("data/pokemon_types.json", "w") as f:
        json.dump(POKEMON_TYPES, f, indent = 8)

    settings.CREATE_TYPES_RECORD = False

def find_alive_pokemon(data: dict, pokemon_details: dict):
    p1_fnt = 0
    p2_fnt = 0
    
    for pokemon in data.get('p1_team_details', []):
        name = pokemon["name"]
        pokemon_details["p1"][name] = pokemon
        pokemon_details["p1"][name]["attack_type_score"] = 0
        pokemon_details["p1"][name]["defense_type_score"] = 0
        pokemon_details["p1"][name]["moves"] = []
        pokemon_details["p1"][name]["state"] = {'status': ['nostatus']}
    
    p2_lead = data["p2_lead_details"]
    pokemon_details["p2"][p2_lead["name"]] = p2_lead
    pokemon_details["p2"][p2_lead["name"]]["attack_type_score"] = 0
    pokemon_details["p2"][p2_lead["name"]]["defense_type_score"] = 0
    pokemon_details["p2"][p2_lead["name"]]["moves"] = []
    pokemon_details["p2"][p2_lead["name"]]["state"] = {'status': ['nostatus']}
    
    for turn in data["battle_timeline"]:
        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]

        # pokemon_details["p1"][p1_name] = { "attack_type_score": 0, "defense_type_score": 0, "moves": [], "state": turn["p1_pokemon_state"] }
        pokemon_details["p1"][p1_name]["attack_type_score"] = 0
        pokemon_details["p1"][p1_name]["defense_type_score"] = 0
        pokemon_details["p1"][p1_name]["moves"] = []
        pokemon_details["p1"][p1_name]["state"] = turn["p1_pokemon_state"]
        # pokemon_details["p2"][p2_name] = { "attack_type_score": 0, "defense_type_score": 0, "moves": [], "state": turn["p2_pokemon_state"] }
        if p2_name not in pokemon_details["p2"]:
            pokemon_details["p2"][p2_name] = {}

        pokemon_details["p2"][p2_name]["attack_type_score"] = 0
        pokemon_details["p2"][p2_name]["defense_type_score"] = 0
        pokemon_details["p2"][p2_name]["moves"] = []
        pokemon_details["p2"][p2_name]["state"] = turn["p2_pokemon_state"]

        if turn["p1_pokemon_state"]["status"] == "fnt":
            del pokemon_details["p1"][p1_name]
            p1_fnt += 1
        if turn["p2_pokemon_state"]["status"] == "fnt":
            del pokemon_details["p2"][p2_name]
            p2_fnt += 1

    for name, p in pokemon_details["p1"].items():
        p["types"] = POKEMON_TYPES[name] if name in POKEMON_TYPES else []
    for name, p in pokemon_details["p2"].items():
        p["types"] = POKEMON_TYPES[name] if name in POKEMON_TYPES else []
    
    return p1_fnt, p2_fnt
        
def find_pokemon_moves(data: dict, pokemon_details: dict):
    for turn in data["battle_timeline"]:
        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]

        p1_move = turn["p1_move_details"]
        p2_move = turn["p2_move_details"]

        if p1_move is not None and p1_name in pokemon_details["p1"] and p1_move["name"] not in pokemon_details["p1"][p1_name]["moves"]:
            pokemon_details["p1"][p1_name]["moves"].append(p1_move)
        if p2_move is not None and p2_name in pokemon_details["p2"] and p2_move["name"] not in pokemon_details["p2"][p2_name]["moves"]:
            pokemon_details["p2"][p2_name]["moves"].append(p2_move)
            
def find_defense_type_score(pokemon_details: dict):
    for p1 in pokemon_details["p1"].values():
        for p2 in pokemon_details["p2"].values():
            for m in p2.get("moves", []):
                if m["category"] == "STATUS":
                    continue
                p1_score = 1
                for _type in p1["types"]:
                    if _type in TABLE_TYPE[m["type"].capitalize()][0]:
                        p1_score *= 0.5
                    if _type in TABLE_TYPE[m["type"].capitalize()][1]:
                        p1_score *= 2
                    if _type in TABLE_TYPE[m["type"].capitalize()][2]:
                        p1_score *= 0

                p1["defense_type_score"] += p1_score
            for m in p1.get("moves", []):
                if m["category"] == "STATUS":
                    continue
                p2_score = 1
                for _type in p2["types"]:
                    if _type in TABLE_TYPE[m["type"].capitalize()][0]:
                        p2_score *= 0.5
                    if _type in TABLE_TYPE[m["type"].capitalize()][1]:
                        p2_score *= 2
                    if _type in TABLE_TYPE[m["type"].capitalize()][2]:
                        p2_score *= 0

                p2["defense_type_score"] += p2_score
                
def find_attack_type_score(pokemon_details: dict):
    for p1 in pokemon_details["p1"].values():
        for p2 in pokemon_details["p2"].values():
            for m in p1.get("moves", []):
                if m["category"] == "STATUS":
                    continue
                p1_score = 1
                for _type in p2["types"]:
                    if _type in TABLE_TYPE[m["type"].capitalize()][0]:
                        p1_score *= 2
                    if _type in TABLE_TYPE[m["type"].capitalize()][1]:
                        p1_score *= 0.5
                    if _type in TABLE_TYPE[m["type"].capitalize()][2]:
                        p1_score *= 0

                p1["attack_type_score"] += p1_score
            for m in p2.get("moves", []):
                if m["category"] == "STATUS":
                    continue
                p2_score = 1
                for _type in p1["types"]:
                    if _type in TABLE_TYPE[m["type"].capitalize()][0]:
                        p2_score *= 2
                    if _type in TABLE_TYPE[m["type"].capitalize()][1]:
                        p2_score *= 0.5
                    if _type in TABLE_TYPE[m["type"].capitalize()][2]:
                        p2_score *= 0

                p2["attack_type_score"] += p2_score

def create_features(data: list[dict]) -> pd.DataFrame:
    load_types_record(data)

    feature_list = []
    for battle in tqdm(data, desc = "Extracting features"):
        features = { }

        # parsing data

        pokemon_details = { "p1": { }, "p2": { } }
        p1_fnt, p2_fnt = find_alive_pokemon(battle, pokemon_details)
        find_pokemon_moves(battle, pokemon_details)
        find_defense_type_score(pokemon_details)
        find_attack_type_score(pokemon_details)

        # creating features

        # number of Pokémon alive
        features["p1_alive"] = len(pokemon_details["p1"])
        features["p2_alive"] = len(pokemon_details["p2"])
                
        # number of Pokémon fainted
        features["p1_fainted"] = p1_fnt
        features["p2_fainted"] = p2_fnt

        # difference between the number of alive Pokémon
        features["p1_p2_alive_diff"] = p1_fnt - p2_fnt

        # number of Pokémon that we don't know anything about
        # features["p2_unknow"] = 6 - len(pokemon_details["p2"])

        # number of Pokémon with a bad status
        features["p1_bad_status"] = len([1 for p in pokemon_details["p1"].values() if p["state"]["status"] not in ["fnt", "nostatus"]])
        features["p2_bad_status"] = len([1 for p in pokemon_details["p2"].values() if p["state"]["status"] not in ["fnt", "nostatus"]])

        # difference between the number of bad status Pokémon
        features["p1_p2_bad_status_diff"] = features["p1_bad_status"] - features["p2_bad_status"]

        # mean hp of the team
        features["p1_mean_hp"] = np.mean([p["state"].get("hp_pct", 1.0) for p in pokemon_details["p1"].values()])
        features["p2_mean_hp"] = np.mean([p["state"].get("hp_pct", 1.0) for p in pokemon_details["p2"].values()])

        # difference between the mean hp of the team
        # features["p1_p2_mean_hp_diff"] = features["p1_mean_hp"] - features["p2_mean_hp"]

        # meam attack of the team (based on moves)
        features["p1_mean_atk"] = np.mean([m["base_power"] * m["accuracy"] for p in pokemon_details["p1"].values() for m in p["moves"] if m["category"] != "STATUS"])
        features["p2_mean_atk"] = np.mean([m["base_power"] * m["accuracy"] for p in pokemon_details["p2"].values() for m in p["moves"] if m["category"] != "STATUS"])

        # features["p1_num_special_moves"] = sum(1 for p in pokemon_details["p1"].values() for m in p["moves"] if m["category"] == "SPECIAL")
        # features["p2_num_special_moves"] = sum(1 for p in pokemon_details["p2"].values() for m in p["moves"] if m["category"] == "SPECIAL")

        # difference between the mean attack
        # features["p1_p2_mean_atk_diff"] = features["p1_mean_atk"] - features["p2_mean_atk"]
        
        # priority moves count
        features["p1_priority_moves"] = sum(1 for p in pokemon_details["p1"].values() for m in p["moves"] if m["priority"] > 0)
        features["p2_priority_moves"] = sum(1 for p in pokemon_details["p2"].values() for m in p["moves"] if m["priority"] > 0)
        
        # num of boosts
        features["p1_total_boosts"] = sum(sum(v for k, v in p["state"].get("boosts", {}).items() if v > 0) for p in pokemon_details["p1"].values())
        features["p2_total_boosts"] = sum(sum(v for k, v in p["state"].get("boosts", {}).items() if v > 0) for p in pokemon_details["p2"].values())
        
        # difference between the num of boosts
        features["p1_p2_total_boosts_diff"] = features["p1_total_boosts"] - features["p2_total_boosts"]

        # mean defense type score of the team
        features["p1_mean_defense_type_score"] = np.mean([p["defense_type_score"] for p in pokemon_details["p1"].values()])
        features["p2_mean_defense_type_score"] = np.mean([p["defense_type_score"] for p in pokemon_details["p2"].values()])

        # difference between the mean defense type score
        features["p1_p2_mean_defense_type_score_diff"] = features["p1_mean_defense_type_score"] - features["p2_mean_defense_type_score"]

        # mean attack type score of the team
        features["p1_mean_atk_type_score"] = np.mean([p["attack_type_score"] for p in pokemon_details["p1"].values()])
        features["p2_mean_atk_type_score"] = np.mean([p["attack_type_score"] for p in pokemon_details["p2"].values()])

        # difference between the mean attack type score
        features["p1_p2_mean_atk_type_score_diff"] = features["p1_mean_atk_type_score"] - features["p2_mean_atk_type_score"]

        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)
