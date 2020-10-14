import json
from pathlib import Path


entities_dir = Path("data/entities")

def json_save(data, fp):
    file_dir = Path(fp).parents[0]
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        json.dump(data, f)
    return

def json_load(fp):
    with open(fp, "r") as f:
        data = json.load(f)
    return data

def load_crypto_entities():
    
    entities = {}
    
    exchanges_fp = entities_dir / "exchanges.json"
    companies_fp = entities_dir / "companies_descriptions.json"
    coin_names_fp = entities_dir / "coin_names.json"

    coins_names_list = []
    coins_names = json_load(coin_names_fp)

    coins_names_list.extend(coins_names.keys())
    coins_names_list.extend(coins_names.values())

    entities["exchanges"] = json_load(exchanges_fp)
    entities["companies"] = json_load(companies_fp)
    entities["coins"] = coins_names_list

    return entities
