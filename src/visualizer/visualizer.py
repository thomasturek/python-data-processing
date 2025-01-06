import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.data_analyzer import TVLAnalyzer

class TVLVisualizer:
    def __init__(self):
        self.analyzer = TVLAnalyzer()

    def _filter_by_timeframe(self, df, days):
        """Helper method to filter DataFrame by number of days."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        end_date = df.index.max()
        start_date = end_date - pd.Timedelta(days=days)
        return df[df.index >= start_date]

    def visualize_data(self):
        try:
            # Initialize analyzer and get data
            _, _, historical_df = self.analyzer.prepare_data()
            predictions_df = self.analyzer.predict_next_days(days=7)
            
            # Get statistical analysis
            stats = self.analyzer.analyze_current_trends(historical_df)
            
            # Ensure consistent DataFrame structure
            historical_df = historical_df.copy()
            if not isinstance(historical_df.index, pd.DatetimeIndex):
                if 'date' in historical_df.columns:
                    historical_df['date'] = pd.to_datetime(historical_df['date'])
                    historical_df.set_index('date', inplace=True)
            
            predictions_df = predictions_df.copy()
            if not isinstance(predictions_df.index, pd.DatetimeIndex):
                predictions_df['date'] = pd.to_datetime(predictions_df['date'])
                predictions_df.set_index('date', inplace=True)
            
            # Normalize predictions to match the last historical value
            last_value = historical_df['tvl'].iloc[-1]
            predictions_df['predicted_tvl'] = np.abs(predictions_df['predicted_tvl'])
            predictions_df['predicted_tvl'] = predictions_df['predicted_tvl'] * (last_value / predictions_df['predicted_tvl'].iloc[0])
            
            print("\nChoose a time period for the data:")
            print("1. Last 7 days + prediction (7d)")
            print("2. Last 1 month + prediction (1m)")
            print("3. All time (no prediction)")
            choice = int(input("Enter the number: "))
            
            # Filter data based on choice
            if choice == 1:
                filtered_historical = self._filter_by_timeframe(historical_df, 7)
                filtered_predictions = predictions_df
                timeframe = '7d'
            elif choice == 2:
                filtered_historical = self._filter_by_timeframe(historical_df, 30)
                filtered_predictions = predictions_df
                timeframe = '30d'
            elif choice == 3:
                filtered_historical = historical_df
                filtered_predictions = None
                timeframe = 'all'
            else:
                filtered_historical = self._filter_by_timeframe(historical_df, 30)
                filtered_predictions = predictions_df
                timeframe = '30d'
                
            print("\nStatistical Indicators Available:")
            print(f"Current TVL: ${stats['current_tvl']/1e9:.2f}B")
            print(f"7-day Average: ${stats['avg_tvl_7d']/1e9:.2f}B")
            print(f"30-day Average: ${stats['avg_tvl_30d']/1e9:.2f}B")
            print(f"7-day Change: {stats['tvl_change_7d']:.2f}%")
            print(f"30-day Change: {stats['tvl_change_30d']:.2f}%")
            
            print("\nChoose statistics to display on plot (separate choices with spaces):")
            print("1. 7-day Average")
            print("2. 30-day Average")
            print("3. None")
            stats_choice = list(map(int, input("Enter the indicators, separated by spaces: ").split()))
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(filtered_historical.index, 
                    filtered_historical['tvl'],
                    label='Historical TVL',
                    color='blue')
            
            # Plot predictions if available
            if filtered_predictions is not None:
                plt.plot(filtered_predictions.index,
                        filtered_predictions['predicted_tvl'],
                        label='Predicted TVL',
                        color='red',
                        linestyle='--')
            
            # Add statistical indicators as horizontal lines
            if 1 in stats_choice:
                plt.axhline(y=stats['avg_tvl_7d'], 
                           color='orange', 
                           linestyle=':', 
                           label='7-day Average')
                
            if 2 in stats_choice:
                plt.axhline(y=stats['avg_tvl_30d'], 
                           color='green', 
                           linestyle=':', 
                           label='30-day Average')
            
            # Format the plot
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            if timeframe == 'all':
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            else:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
                
            plt.gcf().autofmt_xdate()
            plt.xlabel('Date')
            plt.ylabel('TVL Amount')
            title = f'TVL Trends: {timeframe.upper()}'
            if filtered_predictions is not None:
                title += ' with Predictions'
            plt.title(title)
            plt.legend()
            plt.grid(True)
            
            # Set y-axis limits
            max_val = max(
                filtered_historical['tvl'].max(),
                filtered_predictions['predicted_tvl'].max() if filtered_predictions is not None else 0
            )
            plt.ylim(0, max_val * 1.1)
            
            # Format y-axis to billions
            plt.gca().yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
                
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            raise