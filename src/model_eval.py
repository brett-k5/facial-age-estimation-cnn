import os
import json
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model

def in_colab():
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


test_df = pre_processing.test_df
test_df_relevant = pre_processing.test_df_relevant
test_df_13 = pre_processing.test_df_13
test_df_40 = pre_processing.test_df_40

test_ds = pre_processing.test_ds
test_ds_relevant = pre_processing.test_ds_relevant
test_ds_13 = pre_processing.test_ds_13
test_ds_40 = pre_processing.test_ds_40


def save_and_download_json(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    if in_colab():
        files.download(path)


def model_eval(model, test_target, human_estimates_avg, dataset):
    # Save results to JSON in a "metrics" directory
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    if dataset == test_ds or dataset == test_ds_relevant: 
        model_mse, model_mae = model.evaluate(dataset)
        mae_humans = mean_absolute_error(test_target, human_estimates_avg)
        mse_humans = mean_squared_error(test_target, human_estimates_avg)
    
        results = {
            "model_mae": model_mae,
            "model_mse": model_mse,
            "human_mae": mae_humans,
            "human_mse": mse_humans
        }
        print(results)

        if dataset == test_ds:
            results_path = os.path.join(metrics_dir, "model_eval_results.json")
            save_and_download_json(results, results_path)
    
        elif dataset == test_ds_relevant:
            results_path = os.path.join(metrics_dir, "model_eval_relevant.json")
            save_and_download_json(results, results_path)
    
    else:
        predictions_model = model.predict(dataset)
        predictions_model = predictions_model.flatten()

        if dataset == test_ds_13:
            wrong_model = predictions_model >= 21
            wrong_humans = human_estimates_avg >= 21
            results = {
                "Condition": "Predictions >= 21 for real_age <= 13",
                "Num_samples": len(predictions_model),
                "Num_wrong_model": int(np.sum(wrong_model)),
                "Percent_wrong_model": float(np.mean(wrong_model)) * 100,
                "Num_wrong_humans": int(np.sum(wrong_humans)),
                "Percent_wrong_humans": float(np.mean(wrong_humans)) * 100
            }
            print(results)
            results_path = os.path.join(metrics_dir, "model_eval_13.json")
            save_and_download_json(results, results_path)

        else:
            wrong_model = predictions_model < 21
            wrong_humans = human_estimates_avg < 21
            results = {
                "Condition": "Predictions < 21 for real_age >= 40",
                "Num_samples": len(predictions_model),
                "Num_wrong_model": int(np.sum(wrong_model)),
                "Percent_wrong_model": float(np.mean(wrong_model)) * 100,
                "Num_wrong_humans": int(np.sum(wrong_humans)),
                "Percent_wrong_humans": float(np.mean(wrong_humans)) * 100
            }
            print(results)
            results_path = os.path.join(metrics_dir, "model_eval_40.json")
            save_and_download_json(results, results_path)



# Run evaluation
model_eval(model, test_df['real_age'], test_df['apparent_age_avg'], test_ds)

model_eval(model, test_df_relevant['real_age'], test_df_relevant['apparent_age_avg'], test_ds_relevant)

model_eval(model, test_df_13['real_age'], test_df_13['apparent_age_avg'], test_ds_13)

model_eval(model, test_df_40['real_age'], test_df_40['apparent_age_avg'], test_ds_40)