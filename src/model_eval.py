# Scikit-learn (data metrics)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
if in_colab():
    import pre_processing
    from google.colab import files
    uploaded = files.upload()
else:
    from src import pre_processing


test_df = pre_processing.test_df
test_ds = pre_processing.test_ds


# Load the model 
if in_colab():
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})
else:
    model = load_model('models/model.h5')

def model_eval(model, test_target, human_estimates_avg):
    model_mse, model_mae = model.evaluate(test_ds)
    mae_humans = mean_absolute_error(test_target, human_estimates_avg)
    mse_humans = mean_squared_error(test_target, human_estimates_avg)
    print(f"MAE for the model: {model_mae}")
    print(f"MSE for the model: {model_mse}")
    print(f"MAE human guess avg: {mae_humans}")
    print(f"MSE human guess avg: {mse_humans}")

model_eval(model, test_df['real_age'], test_df['apparent_age_avg'])

