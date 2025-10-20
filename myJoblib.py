import joblib

# Function to save a model
def save_model(model, model_name):
    joblib.dump(model, f"{model_name}.pkl")
    print(f"Model saved successfully as {model_name}.pkl")

# Function to load a model
def load_model(model_name):
    return joblib.load(f"{model_name}.pkl")

