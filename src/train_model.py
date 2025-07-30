# Standard library
import os

# Third-party libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if in_colab():
    from google.colab import files
    import shutil
    import pre_processing
else:
    from src import pre_processing

train_ds = pre_processing.train_ds
val_ds = pre_processing.val_ds
test_ds = pre_processing.test_ds
image_shape = pre_processing.image_shape

if in_colab():
    # Define where to save the logs (inside Google Colab environment)
    log_dir = '/content/tensorboard_logs'  # Directory to save TensorBoard logs
    # Ensure the log directory exists, create it if necessary
    os.makedirs(log_dir, exist_ok=True)
else:
    # Define where to save the logs (inside your project directory)
    log_dir = os.path.join(os.getcwd(), 'logs')  # Directory to save TensorBoard logs
    # Ensure the log directory exists, create it if necessary
    os.makedirs(log_dir, exist_ok=True)
    # Logs are saved directly in the project directory
    print(f"Logs saved at: {log_dir}")

# Create a TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5)

def create_model(input_shape):

    """
    It defines model
    """

    backbone = ResNet50(weights='imagenet',
                        input_shape=input_shape,
                        include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model



def train_model(model,
                train_data,
                val_data,
                batch_size=None,
                epochs=20,
                steps_per_epoch=None,
                validation_steps=None):

    """
    Trains the model given the parameters
    """
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)

    if validation_steps is None:
        validation_steps = len(val_data)

    checkpoint = ModelCheckpoint( # Choose the parameters that performed best on validation set
        filepath='best.weights.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True)

    model.fit(train_data,
              validation_data=val_data,
              batch_size=batch_size, 
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              callbacks=[tensorboard_callback],
              verbose=2)

    return model


model = create_model(image_shape)
model = train_model(model, train_ds, val_ds)

if in_colab():
    model.save('model.h5')
    files.download('model.h5')
    # Once training is complete, zip the logs (if you want to download the whole log directory)
    shutil.make_archive(log_dir, 'zip', log_dir)
    # Download the zipped logs file
    files.download(f'{log_dir}.zip')
else:
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
