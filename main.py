from src.analysis.data_analyzer import TVLAnalyzer
from src.visualizer.visualizer import TVLVisualizer

def main():
    try:
        # Initialize analyzer
        analyzer = TVLAnalyzer()
        
        # Get prepared data first
        print("Preparing data...")
        X, y, historical_df = analyzer.prepare_data()
        
        # Train the model
        print("\nTraining model...")
        analyzer.train_model()
        
        # Analyze current trends
        print("\nAnalyzing current trends...")
        trends = analyzer.analyze_current_trends(historical_df)
        print("\nCurrent TVL Trends:")
        for key, value in trends.items():
            print(f"{key}: {value:,.2f}")
        
        # Make predictions
        print("\nGenerating predictions...")
        predictions = analyzer.predict_next_days()
        print("\nPredictions for next 7 days:")
        print(predictions)

        # Visualize the results
        print("\nVisualizing data...")
        visualizer = TVLVisualizer()
        visualizer.visualize_data()

    except ValueError as e:
        print(f"Data validation error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    main()