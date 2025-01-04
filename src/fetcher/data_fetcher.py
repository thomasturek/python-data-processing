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
    
    if response.status_code == 200:
        data = response.json()
        start_date = datetime(2023, 10, 27).date()
        cutoff_date = datetime.now().date()

        if 'chainTvls' in data and 'Ethereum' in data['chainTvls']:
            ethereum_tvl_data = data['chainTvls']['Ethereum']['tvl']
            
            # Filter data by date range and extract relevant fields
            filtered_data = [
                {
                    "date": datetime.fromtimestamp(entry['date']).date(),
                    "tvl": entry['totalLiquidityUSD']
                }
                for entry in ethereum_tvl_data
                if start_date <= datetime.fromtimestamp(entry['date']).date() <= cutoff_date
            ]
            
            # Convert to DataFrame and sort by date
            df = pd.DataFrame(filtered_data)
            if df.empty:
                raise ValueError("No data was retrieved from the API")
                
            return df.sort_values('date').reset_index(drop=True)
    else:
        raise ValueError(f"Error: Received status code {response.status_code}")

    return pd.DataFrame()