# Blockchain TVL Analysis & Prediction System

A sophisticated Python-based analysis tool for predicting and analyzing Total Value Locked (TVL) data from blockchain protocols, specifically focused on Aave. The project implements advanced statistical analysis and deep learning-based predictive analytics using PyTorch, featuring model checkpointing, trend analysis, and interactive data visualization.

## Core Features

The system provides advanced TVL prediction using bidirectional LSTM with attention mechanisms, dual-stream analysis for both trend and volatility prediction, automated model checkpointing and training resumption, interactive data visualization with customizable timeframes, comprehensive statistical analysis of TVL trends, real-time data fetching from DeFi Llama API, and multiple prediction heads for enhanced accuracy.

## Requirements

Python 3.12+, PyTorch 2.2.2+, and all dependencies listed in requirements.txt.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/thomasturek/python-data-processing
cd python-data-processing
```

2. Create and activate a virtual enviroment with

#### Windows

```bash
python -m venv myenv
myenv\Scripts\activate
```

#### Unix/MacOS

```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. Install dependencies with

```bash
pip install -r requirements.txt
```

## Usage

First, activate your virtual environment using the appropriate command for your OS (see above).

Then run the analysis system with various options:

Basic usage with default settings:
```bash
python main.py
```
Force retrain the model from scratch:
```bash
python main.py --force-retrain
```
Customize training parameters:
```bash
python main.py --epochs 200 --learning-rate 0.0005
```
Adjust prediction window:
```bash
python main.py --prediction-days 14
```
## Command Line Arguments

The system accepts several command line arguments: --force-retrain to force the model to retrain from scratch, --epochs to set the number of training epochs (default: 100), --learning-rate to set the learning rate for training (default: 0.001), and --prediction-days to specify the number of days to predict (default: 7).

## Interactive Visualization
The visualization system allows you to select different time periods including last 7 days with prediction, last month with prediction, or all time without prediction. You can also toggle statistical indicators such as 7-day moving average and 30-day moving average, and view current TVL trends and changes.

## Project Structure
The project is organized into several key directories under src/: analysis/ contains the core analysis logic in data_analyzer.py, fetcher/ handles API interaction through data_fetcher.py, models/ defines the PyTorch model architecture in predictor.py, utils/ provides checkpointing utilities in model_utils.py, and visualizer/ implements data visualization in visualizer.py.

## Technical Details

The model architecture employs bidirectional LSTM with attention mechanisms, utilizing dual-stream analysis for trend and volatility. It features multiple prediction heads with weighted loss, gradient clipping, early stopping, and learning rate scheduling.
The training system includes automatic checkpointing of best models, training resumption from checkpoints, validation-based early stopping, adaptive learning rate scheduling, and multiple loss functions for different prediction aspects.

## Libraries Used

The project relies on PyTorch for deep learning, Pandas for data manipulation and analysis, Matplotlib for data visualization, NumPy for numerical computing, Scikit-learn for data preprocessing, and Requests for API interaction.

## Authors

Tomáš Turek
Illya Artemenko
