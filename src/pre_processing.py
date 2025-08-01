# ======================
# Standard Library Imports
# ======================
import os                              # File and path operations
import sys                             # System-specific parameters and functions

# ======================
# Third-Party Imports
# ======================

# Numerical and Data Handling
import numpy as np                     # Numerical computations
import pandas as pd                    # Data handling and analysis

# Progress Bar
from tqdm import tqdm                  # Progress bars

# Scikit-learn (data metrics)
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Deep Learning: TensorFlow + Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Image Processing
from PIL import Image                  # Pillow for image file handling

try:
    from google.colab import drive
    drive.mount('/content/drive')
    os.listdir('/content/drive/MyDrive')
    base_path = '/content/drive/MyDrive'
    train_df = pd.read_csv(os.path.join(base_path, 'gt_avg_train.csv'))
    val_df = pd.read_csv(os.path.join(base_path, 'gt_avg_valid.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'gt_avg_test.csv'))
    print("Loaded data from Google Drive")
except Exception as e:
    print("Not running in Colab or failed to load from Drive:", e)
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

def in_colab():
    return 'google.colab' in sys.modules

if in_colab():
    def get_file_path(folder, fname, use_cropped_faces=True):
        if use_cropped_faces:
            fname = fname.replace('.jpg', '.jpg_face.jpg')
        return os.path.join(base_path, folder, fname)
    train_df['file_path'] = train_df['file_name'].apply(lambda x: get_file_path('train', x))
    val_df['file_path'] = val_df['file_name'].apply(lambda x: get_file_path('valid', x))
    test_df['file_path']  = test_df['file_name'].apply(lambda x: get_file_path('test', x))

else:
    def get_file_path(folder, fname, use_cropped_faces=True):
        if use_cropped_faces:
            fname = fname.replace('.jpg', '.jpg_face.jpg')
        return os.path.join(folder, fname)
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

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)         # Decode image
    image = tf.image.resize(image, [224, 224])              # Resize to uniform size
    image = image / 255.0                                   # Normalize to [0, 1]
    return image, label


def load_data(df, batch_size=32, shuffle=True):

    file_paths = df['file_path'].values
    labels = df['real_age'].values

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Extract image shape from the first batch
    for images, _ in dataset.take(1):
        image_shape = images.shape[1:]  # Drop batch dimension

    return dataset, image_shape

train_ds, image_shape = load_data(train_df, batch_size=32, shuffle=True)
val_ds, _ = load_data(val_df, batch_size=32, shuffle=False)
test_ds, _ = load_data(test_df, batch_size=32, shuffle=False)
test_ds_relevant, _ = load_data(test_df_relevant, batch_size=32, shuffle=False)
test_ds_13, _ = load_data(test_df_13, batch_size=32, shuffle=False)
test_ds_40, _ = load_data(test_df_40, batch_size=32, shuffle=False)