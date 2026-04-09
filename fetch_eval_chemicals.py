# Fetches approved drug names (max_phase=4) from ChEMBL API.
# Used to generate word_list.json for the eval dataset.

import requests
import json

url = "https://www.ebi.ac.uk/chembl/api/data/molecule"

params = {
    "format": "json",
    "limit": 20,
    "max_phase": 4,
    "pref_name__isnull": "false",
}

response = requests.get(url, params=params)
data = response.json()

names = [mol["pref_name"].lower() for mol in data["molecules"] if mol["pref_name"]]

with open("word_list.json", "w") as f:
    json.dump(names, f, indent=2)

print(f"Saved {len(names)} names to word_list.json")
