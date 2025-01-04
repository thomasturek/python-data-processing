from src.analysis.data_analyzer import TVLAnalyzer
from src.visualizer.visualizer import TVLVisualizer

def main():
    try:
        # Initialize analyzer and visualizer
        analyzer = TVLAnalyzer()
        visualizer = TVLVisualizer()

        # Train the model
        print("Training model...")
        analyzer.train_model()
        
        # Get prepared data
        X, y, historical_df = analyzer.prepare_data()
        
        # Analyze current trends
        print("\nCurrent TVL Trends:")
        trends = analyzer.analyze_current_trends(historical_df)
        for key, value in trends.items():
            print(f"{key}: {value:,.2f}")
        
        # Make predictions
        print("\nPredictions for next 7 days:")
        predictions = analyzer.predict_next_days()
        print(predictions)

        # Visualize the results
        print("\nVisualizing data...")
        visualizer.visualize_data()

    except ValueError as e:
        print(f"Data validation error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()