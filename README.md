🧑‍💻 Facial Age Estimation CNN  
Good Seed, a supermarket chain, is exploring whether data science can help improve compliance with alcohol sales regulations — specifically by reducing the risk of selling alcohol to underage individuals. In this project, we construct a convolutional neural network (CNN) to estimate a person's age from facial images, and compare its performance to that of human estimators.

🚀 Project Overview  
This model leverages a ResNet50 backbone for the first 50 layers, followed by:

A convolutional layer utilizing GlobalAveragePooling2D to reduce spatial dimensions

A fully connected dense layer with a single output neuron with a ReLU activation function to predict age

🚀 Training  
We experimented with regularization techniques like Dropout to prevent overfitting, but these did not improve the model's performance significantly.

📊 Performance  
Mean Absolute Error (MAE) on test set: 8.5 years on the full data set 

Near Human level performance on ages 13 to 40 (MAE: 5.52 for the model vs. 4.23 for humans). This is the age range that had the largest volume of data.

The model exhibited significantly poorer performance on ages gelow 13 where the data was sparser. Specifically, it predicted children ages 13 and under were over 21 14.6 percent of the time, compared to just 2.55 percent of the time for humans. 

📊 Instructions for Loading Data and Running Scripts   
The train, valid, and test image sets were too large to be uploaded to github, and they can be accessed at the following google drive links:

train: https://drive.google.com/drive/folders/1ozo04vS91jm7as5AJ1YntpgSdNNB0lc9?usp=drive_link  
valid: https://drive.google.com/drive/folders/162wqDbiJjxsQiHaGuOm2x7YN1NoYsFV5?usp=drive_link  
test: https://drive.google.com/drive/folders/1CSW-DWvmDyfrNBnMBC5l5WInaGnX0xDR?usp=drive_link

If, for some reason, you are struggling to download the data sets from my google drive you can try googling the APPA REAL dataset. As of the time I am writing this you can find it on kaggle here: https://www.kaggle.com/datasets/abhikjha/appa-real-face-cropped

The train, valid, and test image sets should all be moved to the project directory once they are downloaded. As you can see from the pre_processing.py script they are loaded by utilizing their associated file paths which appear in gt_avg_train.csv, gt_avg_valid.csv, and gt_avg_test.csv respectively. 

Each of the .py scripts in the src directory should be run by using python -m src.scrip_name in the terminal of your IDE from the project directory to ensure that they are running from the project directory and properly accessing all of the subdirectories contained in the main project directory. 

🚀 GPU Compatibility  
All .py files in the script are written to be compatible with running in google colab, as are all notebooks, with the exception of the results_and_analysis notebook, which has not code that could conceivably benefit from GPU availability. I included notebooks set up to run the train_model.py script and the model_eval scripts in google colab. These notebooks contain comments to guide you on which files need to be uploaded when using colab's files.upload() function. You can simply upload them to colab and follow the instructions in the comments. 


⚙️ Repository Structure
```
facial_age_estimation_cnn/
├── logs/
│ └── tensorboard_logs.zip # Compressed logs for TensorBoard visualizations
├── metrics/
│ ├── model_eval_13.json # Evaluation metrics for age ≤ 13 subset
│ ├── model_eval_40.json # Evaluation metrics for age ≥ 40 subset
│ ├── model_eval_relevant.json # Evaluation metrics for relevant age subset
│ └── model_eval_results.json # Evaluation metrics for full test set
├── models/
│ └── model.h5 # Saved trained CNN model
├── notebooks/
│ ├── EDA.ipynb
│ ├── model_eval_colab.ipynb
│ ├── results_and_analysis.ipynb
│ └── train_model_colab.ipynb
├── src/
│ ├── init.py
│ ├── model_eval.py
│ ├── pre_processing.py
│ ├── train_model.py
│ └── .vscode/ # Visual Studio Code workspace settings (optional)
│ └── settings.json
├── gt_avg_test.csv
├── gt_avg_train.csv
├── gt_avg_valid.csv
├── README.md
└── requirements.txt