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
from keras.utils import np_utils # Needed for one-hot encoding if not already in data_preprocessor
from sklearn.metrics import confusion_matrix


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


# TrainingData, TestingData = scenarios[user_input] # Already done in the loop

BatchSize=10
NumEpoch=10

# --- START: Corrected Data Loading for A_classes ---
# Load training data for 5-class classification (Normal, A1, A3 in scenario 'a')
X_train, y_train = data_pre.get_processed_data(TrainingData+'.csv', './', classType ='A_classes')

# Load testing data for 5-class classification (A2, A4 in scenario 'a')
# We now correctly load the TestingData file and capture the subclass labels for TPR calculation.
X_test, y_test, subclass_labels_test = data_pre.get_processed_data(
    TestingData+'.csv', 
    './', 
    classType ='A_classes', 
    return_subclass=True
)
# --- END: Corrected Data Loading ---


# ... (Removed commented out and unused code from original)

########################################
# Part 2: Building FNN
#######################################

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train[0])))
#classifier.add(Dropout(0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(0.2))

# --- START: Corrected Output Layer for 5-Class Categorical ---
# Adding the output layer: 5 nodes (Normal, A1, A2, A3, A4)
# 'softmax' ensures the output is a probability distribution over the 5 classes
output_classes = 5 
classifier.add(Dense(units=output_classes, kernel_initializer='uniform', activation='softmax'))

# Compiling the ANN
# Use 'categorical_crossentropy' for multi-class classification
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# --- END: Corrected Output Layer ---

# Fitting the ANN to the Training set
classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch, validation_data=(X_test, y_test))

# evaluate the keras model for the provided model and dataset
loss, accuracy = classifier.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = classifier.evaluate(X_test, y_test, verbose=0)


# Get the final epoch’s training and validation metrics
final_train_acc = classifierHistory.history['accuracy'][-1]
final_train_loss = classifierHistory.history['loss'][-1]
final_val_acc = classifierHistory.history['val_accuracy'][-1]
final_val_loss = classifierHistory.history['val_loss'][-1]

print("\n=== Final Model Performance ===")
print(f"Training Accuracy:  {final_train_acc:.4f}")
print(f"Training Loss:      {final_train_loss:.4f}")
print(f"Validation Accuracy: {final_val_acc:.4f}")
print(f"Validation Loss:    {final_val_loss:.4f}")

########################################
# Part 3 - Making predictions and evaluating the model
#######################################
# Attack class labels corresponding to indices 0-4
# Index 0 is Normal, Index 1 is A1, etc.
A_class_labels = ['Normal', 'A1 (DoS)', 'A2 (Probe)', 'A3 (U2R)', 'A4 (R2L)']

# Predicting the Test set results. y_pred is now a matrix of raw probabilities (5 columns).
y_pred_probabilities = classifier.predict(X_test)

# --- START: Multi-Class Categorization Logic ---
# Convert predictions (probabilities) and true labels (one-hot) to class indices
# np.argmax gets the index of the highest probability, which is the predicted class (0 to 4)
y_pred_classes = np.argmax(y_pred_probabilities, axis=1) 
y_true_classes = np.argmax(y_test, axis=1) # Get the index of the true class
# --- END: Multi-Class Categorization Logic ---

print("\n--- FNN Prediction Output Categorization (A1-A4) ---")

# Example of categorizing the output for the first 10 test samples
print("{:<10} {:<10} {:<15} {:<15}".format("Sample #", "Predicted", "True Label", "Prediction Mapping"))
print("-" * 50)
for i in range(10):
    predicted_index = y_pred_classes[i]
    true_index = y_true_classes[i]
    
    predicted_label = A_class_labels[predicted_index]
    true_label = A_class_labels[true_index]

    # You can now use these labels for reporting or further analysis
    print("{:<10} {:<10} {:<15} {:<15}".format(i+1, predicted_index, true_label, predicted_label))

# Making the Confusion Matrix (5x5)
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix (Rows/Cols: Normal, A1, A2, A3, A4):\n", cm)


# --- START: Updated TPR/SA Calculation for A_classes ---
print ("\ny_test shape:", y_test.shape)
print ("y_pred_probabilities shape:", y_pred_probabilities.shape)
print("unique values in subclass_labels_test:", np.unique(subclass_labels_test))

# The `calculate_tpr_per_attack` function is no longer needed as the logic is complex
# and is handled by the in-line block below for the specific scenario.

if user_input.lower() == 'a':
    print(f"\nSA TPR Calculation for unknown attacks (A2 and A4)")
    
    TPR_A2 = 0.0
    TPR_A4 = 0.0
    
    for subclass in ['A2', 'A4']:
        # 1. Filter for the specific unknown attack (e.g., all samples that are truly 'A2')
        mask = subclass_labels_test == subclass 
        
        # 2. Get the 5-class predictions for ONLY these samples
        y_pred_sub_indices = y_pred_classes[mask] 

        # 3. Collapse the 5-class prediction to binary (Attack vs Normal)
        # Predicted Attack is any index > 0 (A1, A2, A3, A4). Predicted Normal is index 0.
        y_pred_binary = (y_pred_sub_indices != 0).astype(int) 
        
        # 4. True binary label for this filtered set is always 'Attack' (1)
        y_true_binary = np.ones_like(y_pred_binary) 

        # 5. Calculate True Positives (TP) and False Negatives (FN)
        TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        
        TPR = TP / (TP + FN + 1e-10) # 1e-10 for division by zero safety
        
        if subclass == 'A2':
            TPR_A2 = TPR
        elif subclass == 'A4':
            TPR_A4 = TPR

        print(f"TPR for {subclass}: {TPR:.4f}")
    
    avg_tpr_SA = (TPR_A2 + TPR_A4) / 2.0
    print(f"Average TPR for SA: {avg_tpr_SA:.4f}")
    
elif user_input.lower() == 'c':
    print(f"\nSC TPR Calculation for unknown attack A3")
    # This calculation is similar to the one above, but for A3.
    subclass = 'A3'
    mask = subclass_labels_test == subclass
    
    y_pred_sub_indices = y_pred_classes[mask] 
    y_pred_binary = (y_pred_sub_indices != 0).astype(int) 
    y_true_binary = np.ones_like(y_pred_binary) 
    
    TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
    tpr_A3 = TP / (TP + FN + 1e-10)

    print(f"TPR for SC unknown attack A3: {tpr_A3:.4f}")

# --- END: Updated TPR/SA Calculation ---


########################################
# Part 4 - Visualizing
#######################################

# Import matplot lib libraries for plotting the figures. 
import matplotlib.pyplot as plt

history_keys = classifierHistory.history.keys()
#print(f"History keys found: {history_keys}")

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
plt.savefig('train_vs_test_loss.png')
plt.show()
