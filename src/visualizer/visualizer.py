import matplotlib.pyplot as plt
import pandas as pd

from src.fetcher.data_fetcher import get_aave_tvl
from src.analysis.data_analyzer import TVLAnalyzer

class TVLVisualizer:
    def __init__(self):
        self.analyzer = TVLAnalyzer()

    def visualize_data(self):
        try:
            # Get historical data
            historical_df = get_aave_tvl()
            
            if historical_df is None or historical_df.empty:
                print("No historical data available.")
                return

            # Process dates
            if 'date' in historical_df.columns:
                historical_df['date'] = pd.to_datetime(historical_df['date'])
                historical_df.set_index('date', inplace=True)

            # Get trends analysis
            stats = self.analyzer.analyze_current_trends(historical_df)
            print("\nCurrent Trends Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:,.2f}")
                else:
                    print(f"{key}: {value}")

            # Get predictions
            predictions_df = self.analyzer.predict_next_days(7)
            if predictions_df.empty:
                print("No predictions available.")
                return

            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df.set_index('date', inplace=True)

            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(historical_df.index, historical_df['tvl'], 
                    label='Historical TVL', color='blue')
            
            # Plot predictions
            plt.plot(predictions_df.index, predictions_df['predicted_tvl'],
                    label='Predicted TVL', color='orange', linestyle='--')
            
            plt.xlabel('Date')
            plt.ylabel('Total Value Locked (USD)')
            plt.title('Historical TVL and Predicted TVL for AAVE')
            plt.legend()
            plt.grid(True)
            
            # Format y-axis to use billions
            plt.gca().yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B')
            )
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            plt.show()

        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            raise  # Re-raise for debugging