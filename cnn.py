import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, LSTM
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import classification_report
# Function to multiply rows element-wise (as you have defined before)


def Cnn():
    def multiply_rows(matrix):
        matrix = np.array([matrix])
        return np.multiply(matrix, matrix.T)

    # Function to load and preprocess the data for a given CSV file

    def load_and_preprocess_data(csv_file):
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[:, 6:])

        input_data = []
        input_y = []
        window_size = 23
        for i in range(len(scaled_data) - (window_size + 2)):
            templatebox = []
            for j in range(window_size):
                templatebox.append(multiply_rows(scaled_data[i + j, :]))

            input_data.append(templatebox)

            if (scaled_data[i + window_size + 2, 0] - scaled_data[i + window_size, 0]) > 0:
                input_y.append(0)
            else:
                input_y.append(1)

        input_data = np.array(input_data)
        input_y = np.array(input_y)

        return train_test_split(input_data, input_y, test_size=0.2, random_state=42, shuffle=True)

    # Create TensorBoard callback
    tensorboard = TensorBoard(
        log_dir='./logs', write_graph=True, write_images=True)

    # Get the list of all CSV files within the 'data' directory
    data_folder = 'data'
    csv_files = [file for file in os.listdir(
        data_folder) if file.endswith('.csv')]
    print("文件名字加载已经完成.")
    # Train models for each CSV file and save them with different names
    for csv_file in csv_files:  # Use tqdm for the progress bar
        train_data, val_data, train_labels, val_labels = load_and_preprocess_data(
            os.path.join(data_folder, csv_file))
        print(csv_file+"数据预处理已经完成")
        model_file = './cnn_model.h5'
        if os.path.exists(model_file):
            # Load the pre-trained model if it exists
            model = tf.keras.models.load_model(model_file)
        else:
            # Create a new model if it doesn't exist
            model = Sequential([
                Conv2D(256, (2, 2), activation='relu', input_shape=(
                    train_data.shape[1], train_data.shape[2], train_data.shape[3])),

                BatchNormalization(),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.1),
                Dense(32, activation='relu'),
                Dense(2, activation="sigmoid")
            ])
        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # 添加早停回调函数，当验证集上的准确率下降或损失率上升时停止训练
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, mode='min', verbose=1)

        # 训练模型，并将TensorBoard和早停回调函数添加到回调函数列表中
        model.fit(train_data, train_labels, epochs=100, batch_size=32,
                  validation_data=(val_data, val_labels), callbacks=[early_stopping, tensorboard])

        # Save the trained model with a different name based on the CSV file name
        model_filename = f'cnn_model.h5'

        model.save(model_filename)

        # Evaluate on validation data
        val_loss, val_accuracy = model.evaluate(
            val_data, val_labels, verbose=2)
        print(
            f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

        # Get predictions on validation data
        predictions = model.predict(val_data)
        predicted_labels = np.argmax(predictions, axis=1)

        # Print classification report
        print("Classification Report:")
        print(classification_report(val_labels, predicted_labels))


if __name__ == "__main__":
    Cnn()
