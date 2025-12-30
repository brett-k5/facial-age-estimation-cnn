# Standard library imports
import os

# Third-party library imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Local application imports
def in_colab():
    """
    checks to see if script is being run in a google colab environment
    Returns True if it is and False if it isn't
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False
# Change import statement depending on whether or 
if in_colab(): # not script is being run in google colab
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

def create_model(input_shape: tf.TensorShape) -> tf.Keras.Model:
    """
    Builds and compiles a Keras model for regression using a pretrained ResNet50 
    backbone as a feature extractor.

    The model consists of:
        - A ResNet50 base pretrained on ImageNet, with the top classification layer removed
        - A global average pooling layer to flatten feature maps
        - A dense output layer with ReLU activation for regression

    Args:
        input_shape (tf.TensorShape): The shape of a single input image 
                                      (height, width, channels), excluding the batch dimension.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
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



def train_model(model: tf.keras.Model,
                train_data: tf.data.Dataset,
                val_data: tf.data.Dataset,
                batch_size: int = None,
                epochs: int = 20,
                steps_per_epoch: int = None,
                validation_steps: int = None) -> tf.keras.Model:
    """
    Trains a Keras model using the provided training and validation datasets.

    This function handles the training loop with support for checkpointing
    the best model weights based on validation loss, logging training progress,
    and flexible configuration of training parameters such as batch size,
    number of epochs, and steps per epoch.

    Args:
        model (tf.keras.Model): The compiled Keras model to be trained.
        train_data (tf.data.Dataset): Batched training dataset yielding (inputs, targets).
        val_data (tf.data.Dataset): Batched validation dataset for evaluating model
            performance after each epoch.
        batch_size (int, optional): Number of samples per batch. If None, relies on
            the batch size of `train_data`.
        epochs (int, optional): Number of epochs to train the model. Default is 20.
        steps_per_epoch (int, optional): Number of batches to run per training epoch.
            If None, defaults to `len(train_data)`.
        validation_steps (int, optional): Number of batches to run per validation phase.
            If None, defaults to `len(val_data)`.

    Returns:
        tf.keras.Model: The trained model instance with weights updated and best weights
        saved to `'best.weights.h5'`.
    """
    if steps_per_epoch is None: 
        steps_per_epoch = len(train_data) # len(train_data) will be the number of batches per epoch

    if validation_steps is None:
        validation_steps = len(val_data) # len(val_data) will be the number of batches of validation data per epoch

    checkpoint = ModelCheckpoint(# Choose the parameters that performed best on validation set
        filepath='best.weights.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True)

    model.fit(train_data,
              validation_data=val_data,
              batch_size=batch_size, 
              epochs=epochs,
              steps_per_epoch=steps_per_epoch, # one step per batch
              validation_steps=validation_steps, # number of times loss is evaluated per epoch
              callbacks=[tensorboard_callback, checkpoint], # log results
              verbose=2)

    return model


model = create_model(image_shape) # create model
model = train_model(model, train_ds, val_ds) # train model

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
