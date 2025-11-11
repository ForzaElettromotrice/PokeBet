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
def count_status_alter(data: dict) -> Tuple[int, int]:
    p1_pokemons = { }
    p2_pokemons = { }

    for turn in data["battle_timeline"]:
        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]

        p1_pokemons[p1_name] = turn["p1_pokemon_state"]["status"]
        p2_pokemons[p2_name] = turn["p2_pokemon_state"]["status"]

    p1_statuses = 0
    p2_statuses = 0
    for p in p1_pokemons.values():
        if p not in ["fnt", "nostatus"]:
            p1_statuses += 1
    for p in p2_pokemons.values():
        if p not in ["fnt", "nostatus"]:
            p2_statuses += 1

    return p1_statuses, p2_statuses

def calculate_mean_hp(p1_team: dict, p2_team: dict):
    p1_mean = 0
    p2_mean = 0
    
    for p in p1_team.values():
        p1_mean += p.get('hp_pct', 1) # Se non c'è hp_pct significa che non ha perso hp, quindi 100%
    p1_mean /= len(p1_team) if len(p1_team) > 0 else 1

    for p in p2_team.values():
        p2_mean += p.get('hp_pct', 1)
    p2_mean /= len(p2_team) if len(p2_team) > 0 else 1
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

def old_calculate_type_supremacy(data: dict) -> Tuple[int, int]:
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
    p2_points = [[0, 0] for _ in p2_pokemons]
    for i, p1 in enumerate(p1_pokemons.values()):
        for p2 in p2_pokemons.values():
            adv, res = calculate_score(p1, p2)
            p1_points[i][0] += adv
            p1_points[i][1] += res
            
    for i, p2 in enumerate(p2_pokemons.values()):
        for p1 in p1_pokemons.values():
            adv, res = calculate_score(p2, p1)
            p2_points[i][0] += adv
            p2_points[i][1] += res
    return sum(x[0] for x in p1_points), sum(x[1] for x in p1_points), sum(x[0] for x in p2_points), sum(x[1] for x in p2_points)

def normalize_moves(pokemon_moves: dict[str, list[dict]]) -> None:
    for pokemon, moves in pokemon_moves.items():
        for move in moves:
            move["base_power"] *= move["accuracy"]

def calculate_type_supremacy(p1_defense_types: dict[str, set[str]], p1_attack_types: dict[str, list[str]], p2_types: dict[str, list[str]]) -> Tuple[int, int]:
    defense_score = 0
    attack_score = 0
    for types1 in p1_defense_types.values():
        for _type1 in types1:
            _type1 = _type1.capitalize()
            if _type1 == "Notype":
                continue
            for types2 in p2_types.values():
                for _type2 in types2:
                    _type2 = _type2.capitalize()
                    if _type1 in TABLE_TYPE[_type2][0]:
                        defense_score += -1
                    if _type2 in TABLE_TYPE[_type1][1]:
                        defense_score += 1
                    if _type1 in TABLE_TYPE[_type2][2]:
                        defense_score += 2
    for types1 in p1_attack_types.values():
        for _type1 in types1:
            _type1 = _type1.capitalize()
            if _type1 == "Notype":
                continue
            for types2 in p2_types.values():
                for _type2 in types2:
                    if _type2 in TABLE_TYPE[_type1][0]:
                        attack_score += 1
                    if _type2 in TABLE_TYPE[_type1][1]:
                        attack_score += -1
                    if _type1 in TABLE_TYPE[_type2][2]:
                        attack_score += -2         
    return defense_score, attack_score
                     
        
def extract_types(moves: dict[str, list[dict]]) -> dict[str, set[str]]:
    pokemon_types = {}

    for pokemon, moves in moves.items():
        pokemon_types[pokemon] = set()
        for move in moves:
            if move and move.get("type") and move.get("category") != "STATUS":
                pokemon_types[pokemon].add(move["type"].capitalize())

    return pokemon_types
def extract_moves(data: dict) -> Tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    p1_pokemon_moves = { }
    p2_pokemon_moves = { }

    for turn in data["battle_timeline"]:
        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]

        if turn["p1_move_details"] is not None and (p1_name not in p1_pokemon_moves or turn["p1_move_details"] not in p1_pokemon_moves[p1_name]):
            p1_pokemon_moves.setdefault(p1_name, []).append(turn["p1_move_details"])
        if turn["p2_move_details"] is not None and (p2_name not in p2_pokemon_moves or turn["p2_move_details"] not in p2_pokemon_moves[p2_name]):
            p2_pokemon_moves.setdefault(p2_name, []).append(turn["p2_move_details"])

        if turn["p1_pokemon_state"]["status"] == "fnt":
            p1_pokemon_moves.pop(p1_name, None)
        if turn["p2_pokemon_state"]["status"] == "fnt":
            p2_pokemon_moves.pop(p2_name, None)

    return p1_pokemon_moves, p2_pokemon_moves
def find_moves(data: dict):
    p1_pokemon_moves, p2_pokemon_moves = extract_moves(data)
    normalize_moves(p1_pokemon_moves)
    normalize_moves(p2_pokemon_moves)
    p1_pokemon_attack_types = extract_types(p1_pokemon_moves)
    p2_pokemon_attack_types = extract_types(p2_pokemon_moves)
    p1_pokemon_defense_types = {p.get("name"): set(p.get("types")) for p in data.get("p1_team_details")}

    type_defense_score, type_attack_score = calculate_type_supremacy(p1_pokemon_defense_types, p1_pokemon_attack_types, p2_pokemon_attack_types)

    p1_attacks_power = [move["base_power"] for p in p1_pokemon_moves.values() for move in p]
    p2_attacks_power = [move["base_power"] for p in p2_pokemon_moves.values() for move in p]
    
    num_priority_moves_p1 = sum(move.get("priority", 0) for p in p1_pokemon_moves.values() for move in p)
    num_priority_moves_p2 = sum(move.get("priority", 0) for p in p2_pokemon_moves.values() for move in p)
    
    
    
    return np.mean(p1_attacks_power), np.mean(p2_attacks_power), max(p1_attacks_power) if p1_attacks_power != [] else None, max(p2_attacks_power) if p2_attacks_power != [] else None, num_priority_moves_p1, num_priority_moves_p2, type_attack_score, type_defense_score

def team_after_battle(data: dict, p1_team: dict, p2_team: dict) -> Tuple[dict, dict, int, int]:
    p1_fainted = 0
    p2_fainted = 0
    for turn in data["battle_timeline"]:
        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]
        
        if p2_name not in p2_team:
            p2_team[p2_name] = turn["p2_pokemon_state"]

        p1_team[p1_name]['hp_pct'] = turn["p1_pokemon_state"]["hp_pct"]
        p2_team[p2_name]['hp_pct'] = turn["p2_pokemon_state"]["hp_pct"]
        p1_team[p1_name]['status'] = turn["p1_pokemon_state"]["status"]
        p2_team[p2_name]['status'] = turn["p2_pokemon_state"]["status"]
        p1_team[p1_name]['effects'] = turn["p1_pokemon_state"]["effects"]
        p2_team[p2_name]['effects'] = turn["p2_pokemon_state"]["effects"]
        p1_team[p1_name]['boosts'] = turn["p1_pokemon_state"]["boosts"]
        p2_team[p2_name]['boosts'] = turn["p2_pokemon_state"]["boosts"]


        if p1_team[p1_name]['status']  == "fnt":
            del p1_team[p1_name]
            p1_fainted += 1
        if p2_team[p2_name]['status']  == "fnt":
            del p2_team[p2_name]
            p2_fainted += 1

    return p1_team, p2_team, p1_fainted, p2_fainted

def type_multiplier(p1_moves: dict, p2_moves: dict) -> Tuple[float, float]:

    type_pokemon1 = {}
    type_pokemon2 = {}

    for pokemon, moves in p1_moves.items():
        type_pokemon1[pokemon] = []
        for move in moves:
            if move and move.get("type") and move.get("base_power"):
                type_pokemon1[pokemon].append({
                    "type": move["type"].capitalize(),
                    "power": move["base_power"]
                })

    for pokemon, moves in p2_moves.items():
        type_pokemon2[pokemon] = []
        for move in moves:
            if move and move.get("type") and move.get("base_power"):
                type_pokemon2[pokemon].append({
                    "type": move["type"].capitalize(),
                    "power": move["base_power"]
                })

    diz_multiplier_my_pokemon = {}
    diz_multiplier_other_pokemon = {}

    # quanto è efficace il mio pokemon contro i suoi
    for pokemon1, moves1 in type_pokemon1.items():
        total_effectiveness = []

        for pokemon2, moves2 in type_pokemon2.items():
            multiplier = 1.0

            for move in moves1:
                t_att = move["type"]
                base_power = move["power"]

                for t_def_move in moves2:
                    t_def = t_def_move["type"]
                    super_eff, meno_eff, no_eff = TABLE_TYPE[t_att]

                    if t_def in no_eff:
                        multiplier *= 0.0
                    elif t_def in super_eff:
                        multiplier *= 2.0
                    elif t_def in meno_eff:
                        multiplier *= 0.5
                    else:
                        multiplier *= 1.0


                multiplier *= (base_power / 100.0)

            total_effectiveness.append(multiplier)

        diz_multiplier_my_pokemon[pokemon1] = np.mean(total_effectiveness)

    #efficacia dei suoi pokemon contro i miei
    for pokemon2, moves2 in type_pokemon2.items():
        total_effectiveness = []

        for pokemon1, moves1 in type_pokemon1.items():
            multiplier = 1.0

            for move in moves2:
                t_att = move["type"]
                base_power = move["power"]

                for t_def_move in moves1:
                    t_def = t_def_move["type"]
                    super_eff, meno_eff, no_eff = TABLE_TYPE[t_att]

                    if t_def in no_eff:
                        multiplier *= 0.0
                    elif t_def in super_eff:
                        multiplier *= 2.0
                    elif t_def in meno_eff:
                        multiplier *= 0.5
                    else:
                        multiplier *= 1.0

                multiplier *= (base_power / 100.0)

            total_effectiveness.append(multiplier)

        diz_multiplier_other_pokemon[pokemon2] = np.mean(total_effectiveness)

    p1_team_avg = np.mean(list(diz_multiplier_my_pokemon.values()))
    p2_team_avg = np.mean(list(diz_multiplier_other_pokemon.values()))

    return p1_team_avg, p2_team_avg




def create_features(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    for battle in tqdm(data, desc = "Extracting features"):
        features = { }

        # --- Player 1 Team Features ---
        # p1_team = battle.get('p1_team_details', [])
        # if p1_team:
        #     features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
        #     features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
        #     features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
        #     features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
        #     features["p1_mean_spa"] = np.mean([p.get("base_spa", 0) for p in p1_team])
        #     features["p1_mean_spd"] = np.mean([p.get("base_spd", 0) for p in p1_team])
        p1_team = {}
        for p in battle.get('p1_team_details', []):
            p1_team[p['name']] = p
        
        # for i, p in enumerate(p1_team.values()):
        #     features[f'p1_p{i}_base_hp'] = p.get('base_hp', 0)
        #     features[f'p1_p{i}_base_spe'] = p.get('base_spe', 0)
        #     features[f'p1_p{i}_base_atk'] = p.get('base_atk', 0)
        #     features[f'p1_p{i}_base_def'] = p.get('base_def', 0)
        #     features[f'p1_p{i}_base_spa'] = p.get("base_spa", 0)
        #     features[f'p1_p{i}_base_spd'] = p.get("base_spd", 0)
        #     features[f'p1_p{i}_level'] = p.get("level", 0)
        
        p2_team = {}
        p2_lead = battle.get('p2_lead_details', {})
        p2_team[p2_lead.get('name', '')] = p2_lead
        
        # features['p2_lead_base_hp'] = p2_lead.get('base_hp', 0)
        # features['p2_lead_base_spe'] = p2_lead.get('base_spe', 0)
        # features['p2_lead_base_atk'] = p2_lead.get('base_atk', 0)
        # features['p2_lead_base_def'] = p2_lead.get('base_def', 0)
        # features['p2_lead_base_spa'] = p2_lead.get("base_spa", 0)
        # features['p2_lead_base_spd'] = p2_lead.get("base_spd", 0)
        # features['p2_lead_level'] = p2_lead.get("level", 0)
        
        p1_team, p2_team, p1_fnt, p2_fnt = team_after_battle(battle, p1_team, p2_team)
        
        features['p1_team_size'] = len(p1_team)
        features['p2_team_size'] = len(p2_team)

        # p1_fnt, p2_fnt = count_fnt(battle)
        features['p1_fnt'] = p1_fnt
        features['p2_fnt'] = p2_fnt
        features["p1_p2_diff_fnt"] = p1_fnt - p2_fnt

        p1_status_alter, p2_status_alter = count_status_alter(battle)
        features["p1_status_alter"] = p1_status_alter
        features["p2_status_alter"] = p2_status_alter
        features["p1_p2_diff_status_alter"] = p1_status_alter - p2_status_alter

        p1_hps, p2_hps = calculate_mean_hp(p1_team, p2_team)
        features['p1_mean_hp_pct'] = p1_hps
        features['p2_mean_hp_pct'] = p2_hps
        # features["p1_p2_diff_hp_pct"] = p1_hps - p2_hps

        p1_mean_atk, p2_mean_atk, p1_max_atk, p2_max_atk, p1_num_priority, p2_num_priority, type_defense_score, type_attack_score = find_moves(battle)
        features['p1_mean_atk'] = p1_mean_atk
        features['p2_mean_atk'] = p2_mean_atk
        # features['p1_p2_diff_atk'] = p1_mean_atk - p2_mean_atk
        # features['p1_max_atk'] = p1_max_atk
        # features['p2_max_atk'] = p2_max_atk
        # features['p1_num_priority_moves'] = p1_num_priority
        # features['p2_num_priority_moves'] = p2_num_priority
        features["type_attack_score"] = type_attack_score
        features["type_defense_score"] = type_defense_score
        
        p1_adv, p1_res, p2_adv, p2_res = old_calculate_type_supremacy(battle)
        #features['p1_type_adv'] = p1_adv
        #features['p1_type_res'] = p1_res

        #features['p2_type_adv'] = p2_adv
        #features['p2_type_res'] = p2_res
        
        p1_boosts = 0
        for p in p1_team.values():
            for boost in p.get('boosts', {}).values():
                p1_boosts += boost
        features['p1_total_boosts'] = p1_boosts
        p2_boosts = 0
        for p in p2_team.values():
            for boost in p.get('boosts', {}).values():
                p2_boosts += boost
        features['p2_total_boosts'] = p2_boosts
        features['p1_p2_diff_boosts'] = p1_boosts - p2_boosts

        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        #feature_list.append(features)

        # p1_moves, p2_moves = extract_moves(battle)
        # p1_team_avg, p2_team_avg = type_multiplier(p1_moves, p2_moves)
        # features['p1_avg'] = p1_team_avg
        # features['p2_avg'] = p2_team_avg
        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)
