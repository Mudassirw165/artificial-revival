
# revive_predictor.py
# Train ML model to predict cryopreserved cell survival
# Author: Mudassir Waheed, 2025

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cryo_model import simulate_cell_viability

def generate_dataset(samples=500):
    X = []
    y = []
    
    for _ in range(samples):
        # Randomized parameters for simulation
        params = {
            "initial_water_volume": np.random.uniform(0.5, 1.5),
            "initial_solute_concentration": np.random.uniform(0.5, 2.5),
            "cooling_rate": np.random.uniform(-2.0, -0.5),
            "freezing_threshold": np.random.uniform(-15, -5),
            "toxicity_threshold": np.random.uniform(1.5, 3.5),
        }

        result = simulate_cell_viability(params)
        features = [
            params["initial_water_volume"],
            params["initial_solute_concentration"],
            params["cooling_rate"],
            params["freezing_threshold"],
            params["toxicity_threshold"]
        ]

        # Label is 1 if viability ends at 1.0, otherwise 0
        survived = int(result["viability"][-1] > 0.5)
        
        X.append(features)
        y.append(survived)

    return np.array(X), np.array(y)

def train_predictor():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc*100:.2f}%")

    return model

# Train when run directly
if __name__ == "__main__":
    train_predictor()
