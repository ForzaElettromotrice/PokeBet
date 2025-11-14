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

# Tabella calcolata con funzione in utils.py
P_DEF_TYPE= {
        "starmie": [
                "psychic",
                "water"
        ],
        "exeggutor": [
                "grass",
                "psychic"
        ],
        "chansey": [
                "normal"
        ],
        "snorlax": [
                "normal"
        ],
        "tauros": [
                "normal"
        ],
        "alakazam": [
                "psychic"
        ],
        "jynx": [
                "ice",
                "psychic"
        ],
        "slowbro": [
                "psychic",
                "water"
        ],
        "gengar": [
                "ghost",
                "poison"
        ],
        "rhydon": [
                "ground",
                "rock"
        ],
        "zapdos": [
                "electric",
                "flying"
        ],
        "cloyster": [
                "ice",
                "water"
        ],
        "golem": [
                "ground",
                "rock"
        ],
        "jolteon": [
                "electric"
        ],
        "articuno": [
                "flying",
                "ice"
        ],
        "persian": [
                "normal"
        ],
        "lapras": [
                "ice",
                "water"
        ],
        "dragonite": [
                "dragon",
                "flying"
        ],
        "victreebel": [
                "grass",
                "poison"
        ],
        "charizard": [
                "fire",
                "flying"
        ]
}

def team_after_battle(data: dict, p1_team: dict, p2_team: dict) -> Tuple[dict, dict, int, int]:
    """
    Simulate the battle to get the final team states and number of fainted pokemons
    
    Args:
        data (dict): The battle data.
        p1_team (dict): The initial team of player 1.
        p2_team (dict): The initial team of player 2.
        
    Returns:
        p1_team (dict): The final team of player 1.
        p2_team (dict): The final team of player 2.
        p1_fnt (int): The number of fainted pokemons of player 1.
        p2_fnt (int): The number of fainted pokemons of player 2
    """
    p1_fainted = 0
    p2_fainted = 0
    for turn in data["battle_timeline"]:
        p1_name = turn["p1_pokemon_state"]["name"]
        p2_name = turn["p2_pokemon_state"]["name"]
        
        if p2_name not in p2_team:
            p2_team[p2_name] = turn["p2_pokemon_state"]

        # Save the updated state of the pokemon with the keys hp_pct, status, effects, boosts
        for key in turn["p1_pokemon_state"].keys():
            if key == "name":
                continue
            p1_team[p1_name][key] = turn["p1_pokemon_state"][key]
            p2_team[p2_name][key] = turn["p2_pokemon_state"][key]
        
        # Remove the fainted pokemons from the team and increment fainted counter
        if p1_team[p1_name]['status']  == "fnt":
            del p1_team[p1_name]
            p1_fainted += 1
        if p2_team[p2_name]['status']  == "fnt":
            del p2_team[p2_name]
            p2_fainted += 1

    return p1_team, p2_team, p1_fainted, p2_fainted

def count_status_alter(p1_team: dict, p2_team: dict) -> Tuple[int, int]:
    """
    Count the number of pokemons with status alterations (not fainted or no status) for each team.
    
    Args:
        p1_team (dict): The team of player 1.
        p2_team (dict): The team of player 2.
    
    Returns:
        p1_statuses (int): The number of pokemons with status alterations in player 1's team.
        p2_statuses (int): The number of pokemons with status alterations in player 2's team."""
    p1_statuses = 0
    p2_statuses = 0
    
    for pokemom in p1_team.values():
        # if the status is not known, assume no status alteration
        if pokemom.get("status", "nostatus") not in ["fnt", "nostatus"]:
            p1_statuses += 1
    for pokemom in p2_team.values():
        if pokemom.get("status", "nostatus") not in ["fnt", "nostatus"]:
            p2_statuses += 1
    return p1_statuses, p2_statuses

def count_boosts(p1_team: dict, p2_team: dict) -> Tuple[int, int]:
    """
    Count the total number of boosts for each team.
    
    Args:
        p1_team (dict): The team of player
        p2_team (dict): The team of player 2.
        
    Returns:
        p1_boosts (int): The total number of boosts in player 1's team.
        p2_boosts (int): The total number of boosts in player 2's team."""
    p1_boosts = 0
    p2_boosts = 0
    
    for p in p1_team.values():
        for boost in p.get('boosts', {}).values():
            p1_boosts += boost

    for p in p2_team.values():
        for boost in p.get('boosts', {}).values():
            p2_boosts += boost

    return p1_boosts, p2_boosts

def calculate_mean_hp(p1_team: dict, p2_team: dict):
    """
    Calculate the mean hp percentage of the remaining pokemons for each team.
    
    Args:
        p1_team (dict): The team of player 1.
        p2_team (dict): The team of player 2.
    
    Returns:
        p1_mean (float): The mean hp percentage of player 1's team.
        p2_mean (float): The mean hp percentage of player 2's team.
    """
    p1_mean = 0
    p2_mean = 0
    
    for p in p1_team.values():
        p1_mean += p.get('hp_pct', 1) # Se non c'è hp_pct significa che non ha perso hp, quindi 100%
    p1_mean /= len(p1_team) if len(p1_team) > 0 else 1

    for p in p2_team.values():
        p2_mean += p.get('hp_pct', 1)
    p2_mean /= len(p2_team) if len(p2_team) > 0 else 1
    return p1_mean, p2_mean
            
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

def calculate_mean_attack(pokemon_moves: list[dict]) -> float:
    """
    Calculate the mean attack power of the moves used by the pokemons.
    """
    total_power = [move["base_power"] * move["accuracy"] for moves in pokemon_moves.values() for move in moves]
    return np.mean(total_power) if total_power != [] else 0.0

def count_priority_moves(pokemon_moves: list[dict]) -> int:
    """
    Count the number of priority moves used by the pokemons.
    """
    return sum(move.get("priority", 0) for moves in pokemon_moves.values() for move in moves)

def type_multiplier(p1_moves: dict, p2_moves: dict) -> Tuple[float, float]:
    """
    Calculate the average type effectiveness multiplier for each team.
    
    Args:
        p1_moves (dict): The moves used by player 1's pokemons.
        p2_moves (dict): The moves used by player 2's pokemons.
    
    Returns:
        p1_team_avg (float): The average type effectiveness multiplier for player 1's team.
        p2_team_avg (float): The average type effectiveness multiplier for player 2's team.
    """
    type_pokemon1 = {}
    type_pokemon2 = {}

    for pokemon, moves in p1_moves.items():
        type_pokemon1[pokemon] = []
        for move in moves:
            if move and move.get("type") and move.get("base_power") and not move.get("category") == "STATUS":
                type_pokemon1[pokemon].append({
                    "type": move["type"].capitalize(),
                    "power": move["base_power"] * move["accuracy"]
                })

    for pokemon, moves in p2_moves.items():
        type_pokemon2[pokemon] = []
        for move in moves:
            if move and move.get("type") and move.get("base_power") and not move.get("category") == "STATUS":
                type_pokemon2[pokemon].append({
                    "type": move["type"].capitalize(),
                    "power": move["base_power"] * move["accuracy"]
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
                super_eff, meno_eff, no_eff = TABLE_TYPE[t_att]

                for t_def in P_DEF_TYPE.get(pokemon2.lower(), []):
                    # t_def = t_def_move["type"]

                    if t_def in no_eff:
                        multiplier *= 0.0
                    elif t_def in super_eff:
                        multiplier *= 2.0
                    elif t_def in meno_eff:
                        multiplier *= 0.5

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
                super_eff, meno_eff, no_eff = TABLE_TYPE[t_att]

                for t_def in P_DEF_TYPE.get(pokemon1.lower(), []):
                    # t_def = t_def_move["type"]

                    if t_def in no_eff:
                        multiplier *= 0.0
                    elif t_def in super_eff:
                        multiplier *= 2.0
                    elif t_def in meno_eff:
                        multiplier *= 0.5

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

        # Get player 1 initial team
        p1_team = {}
        for p in battle.get('p1_team_details', []):
            p1_team[p['name']] = p
        
        # Get player 2 starting pokemon
        p2_team = {}
        p2_lead = battle.get('p2_lead_details', {})
        p2_team[p2_lead.get('name', '')] = p2_lead
        
        # Simulate the battle to get the final team states and number of fainted pokemons
        p1_team, p2_team, p1_fnt, p2_fnt = team_after_battle(battle, p1_team, p2_team)
        
        # Feature for the size of the teams
        features['p1_team_size'] = len(p1_team)
        features['p2_team_size'] = len(p2_team)

        # Feature for the number of fainted pokemons
        features['p1_fnt'] = p1_fnt
        features['p2_fnt'] = p2_fnt
        features["p1_p2_diff_fnt"] = p1_fnt - p2_fnt

        # Feature for the number of status alterations of the remaining pokemons
        p1_status_alter, p2_status_alter = count_status_alter(p1_team, p2_team)
        features["p1_status_alter"] = p1_status_alter
        features["p2_status_alter"] = p2_status_alter
        #features["p1_p2_diff_status_alter"] = p1_status_alter - p2_status_alter
        
        # Features for the total number of boosts of the remaining pokemons
        p1_boosts, p2_boosts = count_boosts(p1_team, p2_team)
        features['p1_total_boosts'] = p1_boosts
        features['p2_total_boosts'] = p2_boosts
        # features['p1_p2_diff_boosts'] = p1_boosts - p2_boosts

        # Feature for the mean hp percentage of the remaining pokemons
        p1_hps, p2_hps = calculate_mean_hp(p1_team, p2_team)
        features['p1_mean_hp_pct'] = p1_hps
        features['p2_mean_hp_pct'] = p2_hps

        # Get the set of moves used by each player during the battle
        p1_moves, p2_moves = extract_moves(battle)
        
        # Features related to moves
        features['p1_mean_atk'] = calculate_mean_attack(p1_moves)
        features['p2_mean_atk'] = calculate_mean_attack(p2_moves)
        #features['p1_max_atk'] = p1_max_atk
        #features['p2_max_atk'] = p2_max_atk
        features['p1_num_priority_moves'] = count_priority_moves(p1_moves)
        features['p2_num_priority_moves'] = count_priority_moves(p2_moves)
        
        # Features related to type advantages between the pokemons of the two players
        p1_team_avg, p2_team_avg = type_multiplier(p1_moves, p2_moves)
        features['p1_avg'] = p1_team_avg
        features['p2_avg'] = p2_team_avg

        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)
