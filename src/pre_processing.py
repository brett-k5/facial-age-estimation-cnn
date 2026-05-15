"""Create training, validation and testing datasets from file paths to images and labels."""

# Standard library imports
import os
import sys

# Third-party library imports
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm            


def in_colab():
    """Check to see if script is being run in a google colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


if in_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    os.listdir('/content/drive/MyDrive')
    base_path = '/content/drive/MyDrive'
    train_df = pd.read_csv(os.path.join(base_path, 'gt_avg_train.csv'))
    val_df = pd.read_csv(os.path.join(base_path, 'gt_avg_valid.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'gt_avg_test.csv'))
    print("Loaded data from Google Drive")
else:
    try:
        train_df = pd.read_csv('gt_avg_train.csv')
        val_df = pd.read_csv('gt_avg_valid.csv')
        test_df = pd.read_csv('gt_avg_test.csv')
        print("Loaded data from local project directory")
    except Exception as e:
        print("Failed to load CSV files from local directory:", e)
        train_df = None
        val_df = None
        test_df = None

print(train_df.head())
print(val_df.head())
print(test_df.head())
print(len(train_df))
print(len(val_df))
print(len(test_df))


def get_file_path(folder: str, fname: str, use_cropped_faces=True) -> str:
    """Create file path to facial image data.

    Args:
        folder: Folder where data sits.
        fname: Name of the image file.
        use_cropped_faces: Determines whether cropped photos are used. Defaults to True.
    
    Returns:
        A file path to the data.
    """
    if use_cropped_faces:
        fname = fname.replace('.jpg', '.jpg_face.jpg')
    if in_colab:
        return os.path.join(base_path, folder, fname)
    else:
        return os.path.join(folder, fname)


# Apply get_file_path() custom function to train_df, val_df, and test_df utilizing lambda
train_df['file_path'] = train_df['file_name'].apply(lambda x: get_file_path('train', x))
val_df['file_path'] = val_df['file_name'].apply(lambda x: get_file_path('valid', x))
test_df['file_path']  = test_df['file_name'].apply(lambda x: get_file_path('test', x))

# We would like to have a test set that includes specifically the ages we care about too. 
# So let's create one from test_df
test_df_relevant = test_df[(test_df['real_age'] >= 13) & (test_df['real_age'] <=40)] 

# It is okay if our model is significantly wrong for ages <= 13 and >= 40.
# The only reason it would be a problem is if it estimates those ages to be on the wrong side of 21
# We will want to check that too, so let's create two more test_dfs
test_df_13 = test_df[test_df['real_age'] <= 13]
test_df_40 = test_df[test_df['real_age'] >= 40]


def load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Load an image from a file path tensor and appy preprocessing steps.

    Preprocessing steps include:
      - reading the image file from disk
      - decoding it into a tensor
      - resizing it to (224, 224)
      - normalizing pixel values to [0, 1]

    Args:
        path: File path to the image.
        label: Ground truth age label.

    Returns:
            image: Preprocessed image tensor of shape (224, 224, 3), dtype tf.float32.
            label: Ground truth age label. 
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)    # Decode image
    image = tf.image.resize(image, [224, 224])         # Resize to uniform size
    image = image / 255.0                              # Normalize to [0, 1]
    return image, label


def load_data(df: pd.DataFrame, batch_size: int =32, shuffle: bool =True):
    """Load and prepare a TensorFlow dataset from a DataFrame of file paths and labels.

    This function constructs a `tf.data.Dataset` pipeline from a given DataFrame. It performs the following steps:
    1. Extracts file paths and age labels from the DataFrame.
    2. Creates a dataset of (file_path, label) tuples using `from_tensor_slices`.
    3. Optionally shuffles the dataset to eliminate ordering bias.
    4. Applies a preprocessing function to each element in the dataset using `map`.
    5. Batches the dataset according to the specified batch size.
    6. Prefetches batches to improve training performance via pipelining.

    The image shape (excluding batch dimension) is extracted from the first batch
    and returned along with the dataset object.

    Args:
        df: A DataFrame containing 'file_path' and 'real_age' columns.
        batch_size: Number of samples per batch. Default is 32.
        shuffle: Whether to shuffle the dataset. Default is True.

    Returns:
        dataset (tf.data.Dataset): A dataset of preprocessed (image, label) pairs. Done iteratively on an as needed basis. 
        image_shape (tf.TensorShape): The shape of one preprocessed image (H, W, C), excluding batch size.
    """
    file_paths = df['file_path'].values # create 1 dimensional np.ndarray of file paths
    labels = df['real_age'].values # create 1 dimensional np.ndarray of age labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels)) # Lazily slice off tuples from file_paths and labels numpy arrays. 

    if shuffle: 
        # Shuffle the tuples returned by from_tensor_slices so that any unintended order
        dataset = dataset.shuffle(buffer_size=1000) # in the data which might tip off the model will be corrected for

    # Instruct dataset object on how to load and pre_process each (file_path, label) tuple it slices off
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # The below commands execute all of the above dataset commands in order
    # for all samples in batch_size and all prefetched samples in the amount determined by tf.data.AUTOTUNE
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Extract image shape from the first batch
    for images, _ in dataset.take(1):
        image_shape = images.shape[1:]  # Drop batch dimension

    # return samples in data set and image_shape for samples in dataset 
    return dataset, image_shape # This will be done on an as needed basis throughout training


train_ds, image_shape = load_data(train_df, batch_size=32, shuffle=True) # Assign dataset object to train_ds and image shape to image_shape
val_ds, _ = load_data(val_df, batch_size=32, shuffle=False) # We already have the value for our image_shape variable so we just
test_ds, _ = load_data(test_df, batch_size=32, shuffle=False) # assign it to a placeholder variable for the rest of the load_data() calls.


test_ds_relevant, _ = load_data(test_df_relevant, batch_size=32, shuffle=False) # Loads data for ages between 13 and 40 (potential close calls)
test_ds_13, _ = load_data(test_df_13, batch_size=32, shuffle=False) # Loads data for ages 13 and under
test_ds_40, _ = load_data(test_df_40, batch_size=32, shuffle=False) # loads data for ages 40 and up