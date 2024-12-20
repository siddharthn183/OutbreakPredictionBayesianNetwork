import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from google.colab import files
from imblearn.over_sampling import SMOTE
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Upload the data
uploaded = files.upload()
data = pd.read_csv(next(iter(uploaded)))

# Check data
print(data.head())
print(data.info())

# Define the features (X) and target (y)
X = data[['infections', 'newinfections', 'deaths', 'newdeaths', 'temp', 'vaccination_percentage']]
y = data['riskzone']

# Use SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a resampled dataset
data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
data_resampled['riskzone'] = y_resampled

# Check the balance of the 'riskzone' classes
print(data_resampled['riskzone'].value_counts())

# Binning the continuous variables into discrete categories
for col in ['infections', 'newinfections', 'deaths', 'newdeaths', 'temp', 'vaccination_percentage']:
    data_resampled[col] = pd.cut(data_resampled[col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Perform Hill Climbing search to learn the Bayesian Network structure
hc = HillClimbSearch(data_resampled)
best_model = hc.estimate(scoring_method=BicScore(data_resampled))

# Print the learned structure
print("Learned Bayesian Network Structure:")
print(best_model.edges())

# Create a Bayesian Network model based on the learned structure
model = BayesianNetwork(best_model.edges())
model.fit(data_resampled, estimator=BayesianEstimator, prior_type="BDeu")

# Discretization helper function
def discretize_value(value, variable_name):
    if variable_name in ['infections', 'deaths', 'newinfections', 'newdeaths']:
        if value <= 200:
            return 'Very Low'
        elif value <= 400:
            return 'Low'
        elif value <= 600:
            return 'Medium'
        elif value <= 800:
            return 'High'
        else:
            return 'Very High'
    elif variable_name == 'temp':
        if value <= 10:
            return 'Very Low'
        elif value <= 20:
            return 'Low'
        elif value <= 30:
            return 'Medium'
        elif value <= 40:
            return 'High'
        else:
            return 'Very High'
    elif variable_name == 'vaccination_percentage':
        if value <= 20:
            return 'Very Low'
        elif value <= 40:
            return 'Low'
        elif value <= 60:
            return 'Medium'
        elif value <= 80:
            return 'High'
        else:
            return 'Very High'

# Prediction function for risk zone
def predict_risk_zone(infections, newinfections, deaths, newdeaths, temp, vaccination):
    evidence = {
        'infections': infections,
        'newinfections': newinfections,
        'deaths': deaths,
        'newdeaths': newdeaths,
        'temp': temp,
        'vaccination_percentage': vaccination
    }

    inference = VariableElimination(model)
    result = inference.query(variables=['riskzone'], evidence=evidence)

    yellow_prob = result.values[0]
    orange_prob = result.values[1]
    red_prob = result.values[2]

    max_prob = max(yellow_prob, orange_prob, red_prob)
    if max_prob == yellow_prob:
        return 'Yellow', yellow_prob
    elif max_prob == orange_prob:
        return 'Orange', orange_prob
    else:
        return 'Red', red_prob

def get_numeric_input():
    print("Please enter the following information in numeric form:")
    infections = float(input("Number of infections (e.g., 0-1000): "))
    newinfections = float(input("Number of new infections (e.g., 0-1000): "))
    deaths = float(input("Number of deaths (e.g., 0-1000): "))
    newdeaths = float(input("Number of new deaths (e.g., 0-1000): "))
    temp = float(input("Temperature (in °C, e.g., -10 to 50): "))
    vaccination = float(input("Vaccination percentage (0-100): "))

    infections_cat = discretize_value(infections, 'infections')
    newinfections_cat = discretize_value(newinfections, 'newinfections')
    deaths_cat = discretize_value(deaths, 'deaths')
    newdeaths_cat = discretize_value(newdeaths, 'newdeaths')
    temp_cat = discretize_value(temp, 'temp')
    vaccination_cat = discretize_value(vaccination, 'vaccination_percentage')

    return (infections_cat, newinfections_cat, deaths_cat, newdeaths_cat,
            temp_cat, vaccination_cat)

# Visualization of Bayesian Network structure
def visualize_network_structure():
    G = nx.DiGraph(best_model.edges())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
    plt.title("Bayesian Network Structure (Learned via Hill Climbing)")
    plt.axis('off')
    plt.show()

# Model evaluation
def evaluate_model(train_data, test_data):
    # Separate features and target in the training set
    train_X = train_data.drop(columns=['riskzone'])
    train_y = train_data['riskzone']

    # Separate features and target in the testing set
    test_X = test_data.drop(columns=['riskzone'])
    test_y = test_data['riskzone']

    # Fit the model on training data
    model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
    inference = VariableElimination(model)

    # Predict function for test data
    def predict_for_test(data_row):
        evidence = {
            'infections': data_row['infections'],
            'newinfections': data_row['newinfections'],
            'deaths': data_row['deaths'],
            'newdeaths': data_row['newdeaths'],
            'temp': data_row['temp'],
            'vaccination_percentage': data_row['vaccination_percentage']
        }

        result = inference.query(variables=['riskzone'], evidence=evidence)

        # Get the probabilities for each category (Yellow, Orange, Red)
        yellow_prob = result.values[0]
        orange_prob = result.values[1]
        red_prob = result.values[2]

        # Determine the predicted risk zone based on the highest probability
        if yellow_prob > orange_prob and yellow_prob > red_prob:
            return 'Yellow'
        elif orange_prob > yellow_prob and orange_prob > red_prob:
            return 'Orange'
        else:
            return 'Red'

    # Map the actual test labels to the simpler categories ('Yellow', 'Orange', 'Red')
    label_mapping = {
        'Orange Zone': 'Orange',
        'Blue/Yellow Zone': 'Yellow',
        'Red Zone': 'Red'
    }

    # Apply the mapping to the test set
    test_y_mapped = test_y.map(label_mapping)

    # Now apply the prediction function to each row in the test set
    test_predictions = test_X.apply(lambda row: predict_for_test(row), axis=1)

    # Calculate the accuracy using the mapped actual labels
    accuracy = accuracy_score(test_y_mapped, test_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Debug: Compare predictions with the mapped actual values
    print("Predictions vs Mapped Actual Values (First 10):")
    for pred, actual in zip(test_predictions.head(10), test_y_mapped.head(10)):
        print(f"Predicted: {pred}, Actual: {actual}")

# Main execution
def main():
    # Split the resampled data into training and testing sets
    train_data, test_data = train_test_split(data_resampled, test_size=0.2, random_state=42)

    # Visualize the network structure
    visualize_network_structure()

    # Evaluate the model
    evaluate_model(train_data, test_data)

    # Interactive prediction
    print("\n--- Interactive Risk Zone Prediction ---")
    user_infections, user_newinfections, user_deaths, user_newdeaths, user_temp, user_vaccination = get_numeric_input()

    # Predict the risk zone
    predicted_zone, probability = predict_risk_zone(
        infections=user_infections,
        newinfections=user_newinfections,
        deaths=user_deaths,
        newdeaths=user_newdeaths,
        temp=user_temp,
        vaccination=user_vaccination
    )

    # Print the result
    print(f"\nPredicted Risk Zone: {predicted_zone}")
    print(f"Probability: {probability:.2f}")

# Run the main function
if _name_ == "_main_":
    main()