import matplotlib.pyplot as plt
import pandas as pd

from src.fetcher.data_fetcher import get_aave_tvl
from src.analysis.data_analyzer import TVLAnalyzer

class TVLVisualizer:
    def __init__(self):
        self.analyzer = TVLAnalyzer()

    def visualize_data(self):
        historical_data = get_aave_tvl()  
        if historical_data is None:
            print("No historical data fetched (None returned).")
            return
        elif not historical_data:
            print("No historical data fetched (Empty list or dictionary).")
            return

        print("Fetched Data:", historical_data)
        print("Data Type:", type(historical_data))

        if isinstance(historical_data, list):
            if all(isinstance(item, dict) for item in historical_data):
                print("Data format: List of dictionaries - ready to convert to DataFrame")
            elif all(isinstance(item, list) for item in historical_data):
                print("Data format: List of lists - converting to DataFrame using specified columns")
            else:
                print("Unexpected data structure within list")
        else:
            print("Unexpected data type. Expected list of dictionaries or list of lists.")
            return

        try:
            historical_df = pd.DataFrame(historical_data, columns=['date', 'tvl'])
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            historical_df.set_index('date', inplace=True)
        except Exception as e:
            print(f"Error processing historical data into DataFrame: {e}")
            return

        print("Historical DataFrame:")
        print(historical_df)

        if historical_df.empty:
            print("No historical data available after creating DataFrame.")
            return

        stats = self.analyzer.analyze_current_trends(historical_df)
        if not stats:
            print("No trend statistics available.")
            return

        print("Current Trends Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        predictions_df = self.analyzer.predict_next_days(historical_df)
        if predictions_df is None or predictions_df.empty:
            print("No predictions available.")
            return

        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        predictions_df.set_index('date', inplace=True)

        print("Predictions DataFrame:")
        print(predictions_df)

        plt.figure(figsize=(12, 6))
        plt.plot(historical_df.index, historical_df['tvl'], label='Historical TVL', color='blue')
        plt.plot(predictions_df.index, predictions_df['predicted_tvl'], label='Predicted TVL', color='orange', linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel('Total Value Locked (USD)')
        plt.title('Historical TVL and Predicted TVL for AAVE')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    visualizer = TVLVisualizer()
    visualizer.visualize_data()
