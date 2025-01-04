# Blockchain TVL Analysis

A Python-based analysis tool for fetching and analyzing Total Value Locked (TVL) data from various blockchain protocols. The project implements statistical analysis and predictive analytics using PyTorch, displaying results directly in the terminal.

## Core Features

Fetch TVL data from blockchain protocols using ethers
Statistical analysis of TVL trends
Predictive analytics using PyTorch
Terminal-based data visualization
Automated data fetching and caching

## Requirements

Python 3.12
Dependencies are listed in requirements.txt

## Installation

1. Clone the repository with

git clone [(https://github.com/thomasturek/python-data-processing)]
cd analytics-python

2. Create and activate a virtual enviroment with

#### Windows
python -m venv myenv
myenv\Scripts\activate

#### Unix/MacOS
python3 -m venv myenv
source myenv/bin/activate

3. Install dependencies with

pip install -r requirements.txt

## Usage

1. Activate your virtual enviroment with

#### Windows
myenv\Scripts\activate

#### Unix/MacOS
source myenv/bin/activate

2. Run the main analysis and fetch TVL with

python main.py

## Libraries Used

ethers for blockchain interaction and pytorch - for predictive analytics, any additional libraries listed in requirements.txt

## Authors

Tomáš Turek
Illya Artemenko
