import requests
from utils import save_json

def fetch_word_list(limit: int, max_phase: int = 4) -> list[str]:
    """
    Fetches drug preferred names from the ChEMBL API.
    Returns up to limit names with max_phase >= max_phase, lowercased.

    max_phase refers to the highest clinical trial phase a drug has reached,
    where 4 indicates full regulatory approval.

    Paginates automatically if limit exceeds the API's per-page maximum (1000).
    
    Args:
        limit: Maximum number of drug names to fetch
        max_phase: Minimum approval phase to filter by (default 4 = approved drugs only)

    Returns:
        list: lowercased preferred drug names
    """
    
    base_url = "https://www.ebi.ac.uk"
    url = f"{base_url}/chembl/api/data/molecule"

    params: dict[str, str | int] = {
        "format": "json",
        "limit": 1000, # max per page
        "max_phase__gte": max_phase,
        "pref_name__isnull": "false",
    }

    names: list[str] = []

    # Paginate until we have enough names or no more pages
    while len(names) < limit:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        names += [mol["pref_name"].lower() for mol in data["molecules"] if mol["pref_name"]]
        next_page = data["page_meta"]["next"]
        if not next_page:
            break
        url = base_url + next_page
        params = {}

    return names[:limit]


if __name__ == "__main__":
    word_list = fetch_word_list(limit=10000)
    save_json("word_list.json", word_list)
    print(f"Saved {len(word_list)} words to word_list.json")
