import os
from tensorflow.keras.models import load_model
import keras

print(f"Keras version: {keras.__version__}")

# Load the model from the 'models' folder
model = load_model('models/model.h5')

# Get the absolute path to the current script's location
project_dir = os.path.dirname(os.path.abspath(__file__))

# Define the 'models' subdirectory path
models_dir = os.path.join(project_dir, 'models')

# Create the 'models' directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Define full path to save the model
model_path = os.path.join(models_dir, 'model.h5')

# Save the model
model.save(model_path)

print(f"Model saved to: {model_path}")