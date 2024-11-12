from src.analysis.data_analyzer import TVLAnalyzer

def main():
    # Initializing the analyzer
    analyzer = TVLAnalyzer()
    
    # Training the model
    print("Training model...")
    analyzer.train_model()
    
    # Getting current trends
    print("\nCurrent TVL Trends:")
    trends = analyzer.analyze_current_trends()
    for key, value in trends.items():
        print(f"{key}: {value:,.2f}")
    
    # Making predictions for the next 7 days
    print("\nPredictions for next 7 days:")
    predictions = analyzer.predict_next_days()
    print(predictions)

if __name__ == "__main__":
    main()