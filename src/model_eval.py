# Standard library imports
import json
import os
from typing import Any

# Third-party library imports
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model

def in_colab():
    """
    Check to see if script is being run in a colab environment.
    If it is return True, if it isn't return False
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Load data and model based on environment
if in_colab():
    import pre_processing
    from google.colab import files
    uploaded = files.upload()
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})
    base_dir = '/content'  # Colab default working directory
else:
    from src import pre_processing
    model = load_model('models/model.h5')
    base_dir = os.getcwd()


# Assign test DataFrames to variables
test_df = pre_processing.test_df
test_df_relevant = pre_processing.test_df_relevant
test_df_13 = pre_processing.test_df_13
test_df_40 = pre_processing.test_df_40

# Assign test data sets to variables
test_ds = pre_processing.test_ds
test_ds_relevant = pre_processing.test_ds_relevant
test_ds_13 = pre_processing.test_ds_13
test_ds_40 = pre_processing.test_ds_40


def save_and_download_json(results: dict[str, Any], path: str) -> None:
    """
    Saves json file and downloads to device if script is 
    being run in google colab
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    if in_colab():
        files.download(path)


def model_eval(model: tf.keras.Model, test_target: pd.Series, human_estimates_avg: pd.Series, dataset: tf.data.Dataset) -> None:
    # Create a folder to save metrics in
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # measure performance on test sets 
    if dataset == test_ds or dataset == test_ds_relevant: 
        model_mse, model_mae = model.evaluate(dataset)
        mae_humans = mean_absolute_error(test_target, human_estimates_avg)
        mse_humans = mean_squared_error(test_target, human_estimates_avg)
        
        # Create a dictionary to store performance metrics
        results = {
            "model_mae": model_mae,
            "model_mse": model_mse,
            "human_mae": mae_humans,
            "human_mse": mse_humans
        }

        # Print performance metrics
        print(results)
        
        # Save and download results for full test set 
        if dataset == test_ds:
            results_path = os.path.join(metrics_dir, "model_eval_results.json")
            save_and_download_json(results, results_path)
        
        # Save and download results for ages 13 to 40 on the test set
        elif dataset == test_ds_relevant:
            results_path = os.path.join(metrics_dir, "model_eval_relevant.json")
            save_and_download_json(results, results_path)
    
    else:
        # Make predictions for test samples that include only ages 13 and under or only ages 40 and over
        predictions_model = model.predict(dataset)
        predictions_model = predictions_model.flatten()
        
        # Predictions for images of people age 13 and under
        if dataset == test_ds_13:
            wrong_model = predictions_model >= 21 # Calculate how many 13 and under samples the model predicted as over 21
            wrong_humans = human_estimates_avg >= 21 # Calculate how many 13 and under samples humans predicted as over 21

            # Create a dictionary with results
            results = {
                "Condition": "Predictions >= 21 for real_age <= 13",
                "Num_samples": len(predictions_model),
                "Num_wrong_model": int(np.sum(wrong_model)),
                "Percent_wrong_model": float(np.mean(wrong_model)) * 100,
                "Num_wrong_humans": int(np.sum(wrong_humans)),
                "Percent_wrong_humans": float(np.mean(wrong_humans)) * 100
            }
            # Print results
            print(results)

            # Save and download results
            results_path = os.path.join(metrics_dir, "model_eval_13.json")
            save_and_download_json(results, results_path)

        # Predictions for images of people age 40 and over 
        elif dataset == test_ds_21:
            wrong_model = predictions_model < 21
            wrong_humans = human_estimates_avg < 21

            # Create dictionary with results
            results = {
                "Condition": "Predictions < 21 for real_age >= 40",
                "Num_samples": len(predictions_model),
                "Num_wrong_model": int(np.sum(wrong_model)),
                "Percent_wrong_model": float(np.mean(wrong_model)) * 100,
                "Num_wrong_humans": int(np.sum(wrong_humans)),
                "Percent_wrong_humans": float(np.mean(wrong_humans)) * 100
            }
            # Print results
            print(results)

            # Save results
            results_path = os.path.join(metrics_dir, "model_eval_40.json")
            save_and_download_json(results, results_path)
        else:
            raise ValueError("Function received unexpected data set. "
                             "Check to make sure the correct variable was passed to the dataset parameter")



# Run evaluation on relevant test set
model_eval(model, test_df['real_age'], test_df['apparent_age_avg'], test_ds) # Full test set
model_eval(model, test_df_relevant['real_age'], test_df_relevant['apparent_age_avg'], test_ds_relevant) # Ages 13 to 40
model_eval(model, test_df_13['real_age'], test_df_13['apparent_age_avg'], test_ds_13) # Ages 13 and under (measured as binary classification)
model_eval(model, test_df_40['real_age'], test_df_40['apparent_age_avg'], test_ds_40) # Ages 40 and over (measured as binary classification)