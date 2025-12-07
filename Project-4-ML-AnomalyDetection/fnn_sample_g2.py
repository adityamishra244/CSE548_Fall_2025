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
# Import the custom data preprocessor module
import data_preprocessor as data_pre
from sklearn.metrics import confusion_matrix, classification_report # Added classification_report

# Import Keras libraries for building the FNN
from keras.models import Sequential
from keras.layers import Dense, Dropout


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

BatchSize=10
NumEpoch=10

# 🌟 KEY CHANGE 1: Use 'A_classes' to get 5-class labels (Normal, DoS, Probe, U2R, R2L)
# The labels (y_train/y_test) will be One-Hot Encoded vectors of size 5.
X_train, y_train = data_pre.get_processed_data(TrainingData+'.csv', './', classType ='A_classes')
X_test,  y_test, subclass_labels_test  = data_pre.get_processed_data(TestingData+'.csv',  './', classType ='A_classes', return_subclass=True)

# Determine the number of classes (should be 5)
NUM_CLASSES = y_train.shape[1] 
print(f"Dataset loaded with {NUM_CLASSES} classes (Normal, DoS, Probe, U2R, R2L).")


########################################
# Part 2: Building FNN
#######################################

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# input_dim specifies the number of variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train[0])))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# 🌟 KEY CHANGE 2: Output layer for 5 classes (multi-class)
# units = NUM_CLASSES (which is 5)
# activation = 'softmax' for probability distribution over multiple classes
classifier.add(Dense(units = NUM_CLASSES, kernel_initializer = 'uniform', activation = 'softmax'))

# 🌟 KEY CHANGE 3: Compiling the ANN for multi-class classification
# loss = 'categorical_crossentropy' for multi-class problem with One-Hot Encoded labels
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch, validation_data=(X_test, y_test))

# evaluate the keras model for the provided model and dataset
loss, accuracy = classifier.evaluate(X_train, y_train, verbose=0)

# Get the final epoch’s training and validation metrics
final_train_acc = classifierHistory.history['accuracy'][-1]
final_train_loss = classifierHistory.history['loss'][-1]
final_val_acc = classifierHistory.history['val_accuracy'][-1]
final_val_loss = classifierHistory.history['val_loss'][-1]

print("\n=== Final Model Performance (5-Class) ===")
print(f"Training Accuracy:   {final_train_acc:.4f}")
print(f"Training Loss:       {final_train_loss:.4f}")
print(f"Validation Accuracy: {final_val_acc:.4f}")
print(f"Validation Loss:     {final_val_loss:.4f}")

########################################
# Part 3 - Making predictions and evaluating the model
#######################################

# Predicting the Test set results
# y_pred will contain the probability distribution for the 5 classes
y_pred = classifier.predict(X_test)

# Convert One-Hot Encoded y_test and y_pred back to integer class labels for Confusion Matrix
# np.argmax selects the index (0, 1, 2, 3, or 4) with the highest probability
y_true_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Making the Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print('\nPrint the Confusion Matrix (5x5):')
print(cm)

# Define target names for the Classification Report
target_names = ['Normal (0)', 'DoS (A1)', 'Probe (A2)', 'U2R (A3)', 'R2L (A4)']
print("\nClassification Report (5-Class):")
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names, zero_division=0))

########################################
# Part 4 - Visualizing
#######################################

# Import matplot lib libraries for plotting the figures. 
import matplotlib.pyplot as plt

history_keys = classifierHistory.history.keys()
# Determine the correct key names
acc_key = 'accuracy' if 'accuracy' in history_keys else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history_keys else 'val_acc'
loss_key = 'loss' 
val_loss_key = 'val_loss'

# Check if validation data was actually recorded
val_data_present = val_acc_key in classifierHistory.history

print("\nPlotting training performance...")

# --- Accuracy (Train vs Test) ---
plt.figure(figsize=(10, 6))
plt.plot(classifierHistory.history[acc_key], label='Train Accuracy', color='blue', linewidth=2)
if val_data_present:
    plt.plot(classifierHistory.history[val_acc_key], label='Test Accuracy', color='green', linestyle='--', linewidth=2)
plt.title('Training vs Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('train_vs_test_accuracy.png')
plt.show() # 

[Image of Training vs Testing Accuracy Plot]


# --- Loss (Train vs Test) ---
plt.figure(figsize=(10, 6))
plt.plot(classifierHistory.history[loss_key], label='Train Loss', color='red', linewidth=2)
if val_data_present:
    plt.plot(classifierHistory.history[val_loss_key], label='Test Loss', color='orange', linestyle='--', linewidth=2)
plt.title('Training vs Testing Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('train_vs_test_loss.png')
plt.show() # 

[Image of Training vs Testing Loss Plot]


# Print a reminder for the subclass labels
if 'subclass_labels_test' in locals():
    print(f"\nNote: The granular attack labels are stored in 'subclass_labels_test' for detailed analysis.")
