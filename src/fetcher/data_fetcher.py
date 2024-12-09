import requests
import pandas as pd
from datetime import datetime

def get_aave_tvl():
    """
    Fetches Aave TVL data from the DeFi Llama API and returns it as a pandas DataFrame
    """
    url = "https://api.llama.fi/protocol/aave"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise ConnectionError(f"Error: Received status code {response.status_code}")
        
    data = response.json()
    tvl_data = []
    
    if 'chainTvls' in data and 'Ethereum' in data['chainTvls']:
        ethereum_tvl_data = data['chainTvls']['Ethereum']['tvl']
        
        for entry in ethereum_tvl_data:
            date = datetime.fromtimestamp(entry['date']).date()
            tvl_data.append({
                'date': date,
                'tvl': entry['totalLiquidityUSD']
            })
    
    # Convert to DataFrame and sort by date
    df = pd.DataFrame(tvl_data)
    if df.empty:
        raise ValueError("No data was retrieved from the API")
        
    return df.sort_values('date').reset_index(drop=True)