# CNN Model for Time Series Classification

This repository contains a Python script that trains a Convolutional Neural Network (CNN) model for time series classification. The model is trained on multiple CSV files containing time series data and saves the trained models with different names for each input file.

## Prerequisites

- Python 3.6 or higher
- NumPy
- Scikit-learn
- TensorFlow
- Keras

You can install the required dependencies using the following command:

pip install numpy scikit-learn tensorflow keras


## Getting Started

1. Clone this repository to your local machine.
2. Place your CSV files containing time series data in the `data` folder within the repository.
3. Run the Python script `cnn_model.py` to train the CNN models.

## How the Code Works

The Python script `cnn_model.py` follows these main steps:

1. Define a function `multiply_rows(matrix)` to multiply rows element-wise (as defined previously).
2. Define a function `load_and_preprocess_data(csv_file)` to load and preprocess the data for a given CSV file. The data is scaled using Min-Max scaling, and sliding windows are created for each input sequence to prepare the data for CNN training.
3. Load the list of CSV files from the `data` folder and iterate over each file to train the CNN model.
4. The CNN model architecture consists of a 2D Convolutional layer, Batch Normalization, Flatten, and several Dense layers with ReLU activation functions and a final output layer with the sigmoid activation function.
5. The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function.
6. An Early Stopping callback is used to stop training when the validation loss stops improving.
7. The model is trained for 100 epochs on the training data with a batch size of 32.
8. The trained model is saved with the name `cnn_model.h5` in the current directory.
9. The model's performance is evaluated on the validation data, and a classification report is printed.

## Training the Models

To train the CNN models on your data, follow these steps:

1. Ensure your CSV files containing time series data are placed in the `data` folder.
2. Open a terminal or command prompt in the repository directory.
3. Run the following command:

python cnn_model.py


The script will process each CSV file in the `data` folder, train a CNN model for each file, and save the trained models with the name `cnn_model.h5`.

## Model Evaluation

The trained models will be evaluated on the validation data, and the validation loss and accuracy will be printed for each model. Additionally, a classification report will be generated, providing detailed performance metrics for each model.

## Note

If a pre-trained model already exists (saved as `cnn_model.h5`), the script will load the pre-trained model and continue training from the last checkpoint. This behavior is to allow for the continuation of training from a previous session.

To retrain the model from scratch, you can remove the existing `cnn_model.h5` file from the repository.

Feel free to modify the code or adapt it to your specific use case as needed.

If you have any questions or encounter any issues, please don't hesitate to reach out.

# Binance Futures Data Processor and Trading Bot

This Python script is a data processor and trading bot that utilizes the Binance Futures API to fetch and process kline data for specific trading pairs. The data is then used to make trading decisions based on predictions from a pre-trained Convolutional Neural Network (CNN) model. The bot can place market orders for long or short positions on selected trading pairs.

**Disclaimer:** This script involves real money transactions and trading on live accounts. Please use caution and thoroughly understand the code and potential risks before running it on a live account. It is advisable to test the bot on a demo account first to ensure its behavior aligns with your trading strategy.

## Prerequisites

1. Python 3.6 or higher
2. Required Python libraries: numpy, pandas, requests, mysql-connector-python, scikit-learn, keras, and binance

Install the required dependencies using the following command:

```bash
pip install numpy pandas requests mysql-connector-python scikit-learn keras python-binance

