from src.analysis.data_analyzer import TVLAnalyzer
from src.visualizer.visualizer import TVLVisualizer
import argparse

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='TVL Analysis and Prediction System')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining the model from scratch')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--prediction-days', type=int, default=7,
                       help='Number of days to predict into the future')
    args = parser.parse_args()

    try:
        # Initialize analyzer
        print("\n=== TVL Analysis and Prediction System ===")
        analyzer = TVLAnalyzer()
        
        # Get prepared data and train model
        print("\n[1/3] Preparing data and training model...")
        X, y, historical_df = analyzer.prepare_data()
        analyzer.train_model(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            force_retrain=args.force_retrain
        )
        
        # Generate predictions
        print(f"\n[2/3] Generating predictions for next {args.prediction_days} days...")
        analyzer.predict_next_days(days=args.prediction_days)

        # Visualize the results
        print("\n[3/3] Visualizing data...")
        visualizer = TVLVisualizer()
        visualizer.visualize_data()
        
        print("\n✓ Analysis completed successfully!")

    except ValueError as e:
        print(f"\n❌ Data validation error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()