# IMPORT AND INSTALL LIBRARIES

# #tensorflow==2.5.1
# tensorflow-gpu==2.5.1
# scikit-learn

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import keras

#------------------------------------------------------------------------------

# Function to load the data from the file and store into sequence[] and labels[]

def load_data():
    sequences = []
    labels = []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)

    return X, Y

#------------------------------------------------------------------------------

# Function to create the model and train the model

def create_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model and save the model if the categorical_accuracy is greater than 92% and epochs are more than 170

    for i in range (2000):
        model.fit(X_train, y_train, epochs=1, callbacks=[tb_callback])

        if model.history.history['categorical_accuracy'][0] > 0.92 and i > 170:
            model.save('action.h5')
            break

    return model

#     model = Sequential()
#     model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
#     model.add(LSTM(128, return_sequences=True, activation='relu'))
#     model.add(LSTM(64, return_sequences=False, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(actions.shape[0], activation='softmax'))

#     model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#    # Define EarlyStopping callback
#     early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=5, mode='max', min_delta=0.001, restore_best_weights=True)

#     # Custom stopping criteria
#     class CustomStopCallback(keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs=None):
#             if logs.get('val_categorical_accuracy') > 0.92 and epoch > 170:
#                 self.model.stop_training = True

#     # Create an instance of the custom callback
#     custom_stop_callback = CustomStopCallback()

#     # Train the model
#     model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[tb_callback, early_stopping, custom_stop_callback])

#     # Save the model
#     model.save('action.h5')

#     return model

#------------------------------------------------------------------------------

# Setup the parameters for the model and train the model

actions = np.array(['hello', 'thanks', 'jayShreeRam'])

DATA_PATH = os.path.join('MP_Data')

no_sequences = 30

sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

log_dir = os.path.join('Logs')

tb_callback = TensorBoard(log_dir=log_dir)

#------------------------------------------------------------------------------

# Load the data

X, Y = load_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

#------------------------------------------------------------------------------

# Create the model and train the model

model = create_model(X_train, Y_train)

#------------------------------------------------------------------------------