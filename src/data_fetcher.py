import requests
from datetime import datetime

def get_aave_tvl():
    url = "https://api.llama.fi/protocol/aave"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        start_date = datetime(2023,10,27).date()
        cutoff_date = datetime.now().date()
        if 'chainTvls' in data and 'Ethereum' in data['chainTvls']:
            ethereum_tvl_data = data['chainTvls']['Ethereum']['tvl']
            for entry in ethereum_tvl_data:
                entry_date = datetime.fromtimestamp(entry['date']).date()
                if start_date <= entry_date <= cutoff_date:
                    print(f"Date: {entry_date}, Total Liquidity (USD): {entry['totalLiquidityUSD']}")
    else:
        print(f"Error: Received status code {response.status_code}")

get_aave_tvl()