from src.analysis.data_analyzer import TVLAnalyzer
from src.visualizer.visualizer import TVLVisualizer
from src.fetcher.data_fetcher import get_aave_tvl
import pandas as pd

def main():
    try:
        # Initializing the analyzer
        analyzer = TVLAnalyzer()
        
        # Get historical data
        historical_data = get_aave_tvl()
        if not historical_data:
            print("Error: No historical data available")
            return
            
        # Process historical data
        historical_df = pd.DataFrame(historical_data)
        if 'date' not in historical_df.columns or 'tvl' not in historical_df.columns:
            print("Error: Missing required columns in historical data")
            return
            
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        
        # Training the model
        print("Training model...")
        analyzer.train_model()
        
        # Getting current trends
        print("\nCurrent TVL Trends:")
        trends = analyzer.analyze_current_trends(historical_df)  # Pass the historical data
        for key, value in trends.items():
            print(f"{key}: {value:,.2f}")
        
        # Making predictions for the next 7 days
        print("\nPredictions for next 7 days:")
        predictions = analyzer.predict_next_days()
        print(predictions)

        # Visualize data
        print("\nVisualizing data...")
        visualizer = TVLVisualizer()
        visualizer.visualize_data()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()