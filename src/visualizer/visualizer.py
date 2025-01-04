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

        try:
            historical_df = pd.DataFrame(historical_data, columns=['date', 'tvl'])
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            historical_df.set_index('date', inplace=True)
        except Exception as e:
            print(f"Error processing historical data into DataFrame: {e}")
            return

        if historical_df.empty:
            print("No historical data available after creating DataFrame.")
            return

        try:
            stats = self.analyzer.analyze_current_trends(historical_df)
            if not stats:
                print("No trend statistics available.")
                return

            print("Current Trends Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")

            predictions_df = self.analyzer.predict_next_days(7)  # Specify number of days explicitly
            if predictions_df is None or predictions_df.empty:
                print("No predictions available.")
                return

            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df.set_index('date', inplace=True)

            plt.figure(figsize=(12, 6))
            plt.plot(historical_df.index, historical_df['tvl'], label='Historical TVL', color='blue')
            plt.plot(predictions_df.index, predictions_df['predicted_tvl'], label='Predicted TVL', color='orange', linestyle='--')
            
            plt.xlabel('Date')
            plt.ylabel('Total Value Locked (USD)')
            plt.title('Historical TVL and Predicted TVL for AAVE')
            plt.legend()
            plt.grid()
            plt.show()
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            return

if __name__ == "__main__":
    visualizer = TVLVisualizer()
    visualizer.visualize_data()
