# Pandemic Prediction Using Bayesian Networks

This project predicts pandemic risk zones (e.g., Blue/Yellow, Orange, Red) using a Bayesian Network model. The model leverages epidemiological data (e.g., infections, deaths) and environmental factors (e.g., temperature, vaccination rates) to provide a probabilistic prediction of the pandemic threat level.

## Project Overview

Pandemic risk prediction has been crucial for informed decision-making during health crises like COVID-19. This model uses a Bayesian Network to infer risk zones by learning dependencies between infection rates, mortality, vaccination, and temperature. The model is trained on real-world data from Germany, which includes features relevant to tracking and predicting pandemic zones.

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Example Results](#example-results)

## Dataset

The dataset contains data of various locations. Each record has attributes such as:
- **Infections and Deaths:** Current COVID-19 cases and deaths in each state.
- **Date:** Date of the recorded data.
- **New Infections and Deaths:** Daily increments of cases and fatalities.
- **Temperature:** Average temperature, potentially affecting the virus spread.
- **Vaccination Percentage:** Population vaccination rate by state.
- **Risk Zone:** The pandemic threat level, our target variable.

## Model

The Bayesian Network was built using the `pgmpy` library to model dependencies among features. Important dependencies were identified based on domain knowledge, such as:
- **Infections and Deaths:** Direct influence on risk zones.
- **Temperature and Vaccination Percentage:** Indirect influence on risk due to seasonality and immunity.
- **New Infections and Deaths:** Dynamic indicators of risk trends.

Using the Variable Elimination inference method, the model predicts risk zones based on current state inputs, giving a probability distribution over the possible zones.

## Dependencies

Install the necessary libraries before running the code:
```bash
pip install pandas pgmpy scikit-learn imbalanced-learn

Usage
  Running the Code
  Clone the repository:
      git clone https://github.com/siddharthn183/OutbreakPredictionBayesianNetwork.git
      cd OutbreakPredictionBayesianNetwork

Example Results
When provided with the following input:
  Infections: 1000
  Deaths: 250
  Temperature: 25C
  Vaccination Percentage: 90%
The model may output:
  Predicted Risk Zone: Orange Zone
  Probability Distribution:
    - Blue/Yellow Zone: 5%
    - Orange Zone: 65%
    - Red Zone: 30%

