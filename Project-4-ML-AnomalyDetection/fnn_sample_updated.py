# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:43:04 2019

Updated on Wed Jan 29 10:18:09 2020

@author: created by Sowmya Myneni and updated by Dijiang Huang
"""

########################################
# Part 1 - Data Pre-Processing
#######################################

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np
#import tensorflow as tf
import data_preprocessor as data_pre


#===Updated code to process all 3 Testing Scenarios======
scenarios = {
    'a': ['Training-a1-a3', 'Testing-a2-a4'],
    'b': ['Training-a1-a2', 'Testing-a1'],
    'c': ['Training-a1-a2', 'Testing-a1-a2-a3']
}

print('Please enter the scenario you wish to run: either a, b, or c')
for key in scenarios.keys():
    print(f'  - {key}, or {key.upper()}')

# Loop until user enters a valid choice
while True:
    user_input = input('Enter your choice: ').lower().strip()
    if user_input in scenarios:
        TrainingData, TestingData = scenarios[user_input]
        break
    else:
        print('Invalid scenario. Please try again.')

# Display the selected datasets
print(f"\nYou selected scenario '{user_input.upper()}'.")
print(f"Training data: {TrainingData}")
print(f"Testing data: {TestingData} \n")


TrainingData, TestingData = scenarios[user_input]

'''
# Assign the selected training and testing dataset names
if scenario == 'a':
    TrainingData = scenarios['a'][0]
    TestingData = scenarios['a'][1]
elif scenario == 'b':
    TrainingData = scenarios['b'][0]
    TestingData = scenarios['b'][1]
elif scenario == 'c':
    TrainingData = scenarios['c'][0]
    TestingData = scenarios['c'][1]
'''

BatchSize=10
#BatchSize=15
#BatchSize=20
#NumEpoch=10
NumEpoch=10
#NumEpoch=20


X_train, y_train = data_pre.get_processed_data(TrainingData+'.csv', './', classType ='binary')
X_test,  y_test  = data_pre.get_processed_data(TestingData+'.csv',  './', classType ='binary')

#tf.autograph.set_verbosity(0)
#@tf.autograph.experimental.do_not_convert

#========================================================

# Variable Setup
# Available datasets: KDDTrain+.txt, KDDTest+.txt, etc. More read Data Set Introduction.html within the NSL-KDD dataset folder
# Type the training dataset file name in ''
#TrainingDataPath='NSL-KDD/'
#TrainingData='KDDTrain+_20Percent.txt'
#TrainingData='KDDTrain+.txt'
#TrainingData='KDDTest+.txt'
#TrainingData='KDDTest-21.txt'
# Batch Size
#BatchSize=10
# Epohe Size
#NumEpoch=10


# Import dataset.
# Dataset is given in TraningData variable You can replace it with the file 
# path such as “C:\Users\...\dataset.csv’. 
# The file can be a .txt as well. 
# If the dataset file has header, then keep header=0 otherwise use header=none
# reference: https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
'''
dataset = pd.read_csv(TrainingDataPath+TrainingData, header=None)
X = dataset.iloc[:, 0:-2].values
label_column = dataset.iloc[:, -2].values
y = []
for i in range(len(label_column)):
    if label_column[i] == 'normal':
        y.append(0)
    else:
        y.append(1)

# Convert ist to array
y = np.array(y)
'''

# Encoding categorical data (convert letters/words in numbers)
# Reference: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
# The following code work without warning in Python 3.6 or older. Newer versions suggest to use ColumnTransformer
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [1, 2, 3])
X = onehotencoder.fit_transform(X).toarray()
'''
# The following code work Python 3.7 or newer

'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1,2,3])],    # The column numbers to be transformed ([1, 2, 3] represents three columns to be transferred)
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)
'''

# Splitting the dataset into the Training set and Test set (75% of data are used for training)
# reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Perform feature scaling. For ANN you can use StandardScaler, for RNNs recommended is 
# MinMaxScaler. 
# referece: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/preprocessing.html
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Scaling to the range [0,1]
X_test = sc.fit_transform(X_test)
'''

########################################
# Part 2: Building FNN
#######################################

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
# Reference: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
classifier = Sequential()

# Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables
# rectified linear unit activation function relu, reference: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
classifier.add(Dropout(0.2))

# Adding the second hidden layer, 6 nodes
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the output layer, 1 node, 
# sigmoid on the output layer is to ensure the network output is between 0 and 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN, 
# Gradient descent algorithm “adam“, Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“, Reference: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.
# add verbose=0 to turn off the progress report during the training
# To run the whole training dataset as one Batch, assign batch size: BatchSize=X_train.shape[0]
classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch, validation_data=(X_test, y_test))

# evaluate the keras model for the provided model and dataset
loss, accuracy = classifier.evaluate(X_train, y_train)
test_loss, test_accuracy = classifier.evaluate(X_test, y_test)
#print('Print the Training loss and the accuracy of the model on the dataset')
#print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))
#print('Print the testing loss and the accuracy of the model on the dataset')
#print('Loss [0,1]: %.4f' % (test_loss), 'Accuracy [0,1]: %.4f' % (test_accuracy))

# Get the final epoch’s training and validation metrics
final_train_acc = classifierHistory.history['accuracy'][-1]
final_train_loss = classifierHistory.history['loss'][-1]
final_val_acc = classifierHistory.history['val_accuracy'][-1]
final_val_loss = classifierHistory.history['val_loss'][-1]

print("\n=== Final Model Performance ===")
print(f"Training Accuracy:  {final_train_acc:.4f}")
print(f"Training Loss:      {final_train_loss:.4f}")
print(f"Validation Accuracy: {final_val_acc:.4f}")
print(f"Validation Loss:     {final_val_loss:.4f}")

########################################
# Part 3 - Making predictions and evaluating the model
#######################################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.9)   # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9
y_pred = (y_pred > 0.5).astype(int)
# summarize the first 5 cases
#for i in range(5):
#	print('%s => %d (expected %d)' % (X_test[i].tolist(), y_pred[i], y_test[i]))

# Making the Confusion Matrix
# [TN, FP ]
# [FN, TP ]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)

#TN = cm[0, 0]
#FP = cm[0, 1]
#FN = cm[1, 0]
#TP = cm[1, 1]

#total_samples = TP + TN + FP + FN
#accuracy = (TP + TN) / total_samples

#print(f"\nCalculated Testing Accuracy: {accuracy:.4f}")
########################################
# Part 4 - Visualizing
#######################################

# Import matplot lib libraries for plotting the figures. 
import matplotlib.pyplot as plt

'''
# You can plot the accuracy
print('Plot the accuracy')
# Keras 2.2.4 recognizes 'acc' and 2.3.1 recognizes 'accuracy'
# use the command python -c 'import keras; print(keras.__version__)' on MAC or Linux to check Keras' version
plt.plot(classifierHistory.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('accuracy_sample.png')
plt.show()

# You can plot history for loss
print('Plot the loss')
plt.plot(classifierHistory.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('loss_sample.png')
plt.show()
'''

# Keras history keys can be 'accuracy'/'val_accuracy' or 'acc'/'val_acc' 
# depending on Keras version/TensorFlow settings.
history_keys = classifierHistory.history.keys()
print(f"History keys found: {history_keys}")

# Determine the correct key names
if 'accuracy' in history_keys:
    acc_key = 'accuracy'
    val_acc_key = 'val_accuracy'
    loss_key = 'loss'
    val_loss_key = 'val_loss'
elif 'acc' in history_keys:
    acc_key = 'acc'
    val_acc_key = 'val_acc'
    loss_key = 'loss'
    val_loss_key = 'val_loss'
else:
    print("Error: Could not find standard accuracy/loss keys in history. Check Keras version.")
    # Fallback to single line if validation data wasn't recorded properly
    acc_key = 'accuracy'
    loss_key = 'loss'
    val_acc_key = None # Marker that validation data isn't present

########################################
# 1️⃣ Training-only plots
########################################
print("\nPlotting training-only accuracy and loss...")
plt.figure()
plt.plot(classifierHistory.history[acc_key], color='blue', linewidth=2)
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.savefig('train_accuracy_only.png')
plt.show()

plt.figure()
plt.plot(classifierHistory.history[loss_key], color='red', linewidth=2)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.savefig('train_loss_only.png')
plt.show()

########################################
# 2️⃣ Testing-only plots (if available)
########################################
if val_acc_key in classifierHistory.history:
    print("\nPlotting testing-only (validation) accuracy and loss...")
    plt.figure()
    plt.plot(classifierHistory.history[val_acc_key], color='green', linewidth=2)
    plt.title('Validation (Testing) Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.savefig('test_accuracy_only.png')
    plt.show()

    plt.figure()
    plt.plot(classifierHistory.history[val_loss_key], color='orange', linewidth=2)
    plt.title('Validation (Testing) Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.savefig('test_loss_only.png')
    plt.show()

########################################
# 3️⃣ Combined plots (train vs test)
########################################
print("\nPlotting combined train vs test accuracy and loss...")

# --- Accuracy (Train vs Test) ---
plt.figure()
plt.plot(classifierHistory.history[acc_key], label='Train Accuracy', color='blue', linewidth=2)
if val_acc_key:
    plt.plot(classifierHistory.history[val_acc_key], label='Test Accuracy', color='green', linestyle='--', linewidth=2)
plt.title('Training vs Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('train_vs_test_accuracy.png')
plt.show()

# --- Loss (Train vs Test) ---
plt.figure()
plt.plot(classifierHistory.history[loss_key], label='Train Loss', color='red', linewidth=2)
if val_loss_key:
    plt.plot(classifierHistory.history[val_loss_key], label='Test Loss', color='orange', linestyle='--', linewidth=2)
plt.title('Training vs Testing Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
