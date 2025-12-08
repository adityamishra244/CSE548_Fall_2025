# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:36:17 2020

@author: created by Sowmya Myneni and updated by Dijiang Huang
"""
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#=============NEW: Import the attack class mappings from  distinctLabelExtractor class================**
try:
    from distinctLabelExtractor import attacks_subClass, expectedAttackClasses
except ImportError:
    # Fallback: Define the lists if import fails, using the logic found in distinctLabelExtractor.py
    # A1=DoS (index 0), A2=Probe (index 1), A3=U2R (index 2), A4=R2L (index 3)
    attacks_subClass = [['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'],
        ['ipsweep', 'mscan', 'portsweep', 'saint', 'satan','nmap'],
        ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'],
        ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
        ]
    expectedAttackClasses = ['DoS (A1)', 'Probe (A2)', 'U2R (A3)', 'R2L (A4)']
#========================================================================================================#    
#========================================Multi Class Processing==========================================#
def map_attack_to_A_class(attack_name):
    """Maps an attack name to its categorical index: 0=Normal, 1=A1, 2=A2, 3=A3, 4=A4"""
    lower_name = str.lower(str(attack_name))
    if lower_name == 'normal':
        return 0  # Class 0: Normal

    # Iterate through the 4 attack subclasses (A1, A2, A3, A4)
    for i, attack_list in enumerate(attacks_subClass):
        if lower_name in attack_list:
            return i + 1  # Class 1, 2, 3, or 4

    # If an attack is not found in the list (e.g., a new attack type)
    return -1 # Use -1 to indicate unknown/unhandled attacks
#========================================================================================================#

def get_processed_data(datasetFile, categoryMappingsPath, classType='binary', return_subclass=False):
    inputFile = pd.read_csv(datasetFile, header=None)
    X = inputFile.iloc[:, 0:-2].values
    label_column = inputFile.iloc[:, -2].values
    #subclass_column = inputFile.iloc[:, -1].values
    
    category_1 = np.array(pd.read_csv(categoryMappingsPath + "1.csv", header=None).iloc[:, 0].values)
    category_2 = np.array(pd.read_csv(categoryMappingsPath + "2.csv", header=None).iloc[:, 0].values)
    category_3 = np.array(pd.read_csv(categoryMappingsPath + "3.csv", header=None).iloc[:, 0].values)
    #category_label = np.array(pd.read_csv(categoryMappingsPath + "41.csv", header=None).iloc[:, 0].values)
    ct = ColumnTransformer(
                [('X_one_hot_encoder', OneHotEncoder(categories=[category_1, category_2, category_3], handle_unknown='ignore'), [1,2,3])],    # The column numbers to be transformed ([1, 2, 3] represents three columns to be transferred)
                remainder='passthrough'# Leave the rest of the columns untouched
            )
    X = np.array(ct.fit_transform(X), dtype=np.float)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(np.array(X))  # Scaling to the range [0,1]
        
    if classType == 'binary':               
        y = []
        for i in range(len(label_column)):
            if label_column[i] == 'normal' or str(label_column[i]) == '0':
                y.append(0)
            else:
                y.append(1)        
        # Convert ist to array
        y = np.array(y)    
    elif classType == 'A_classes': # <--- NEW classType
        y_a_classes = [map_attack_to_A_class(label) for label in label_column]
    
        # Filter out unhandled/unknown attacks if necessary, or assign them to 'Normal' (0)
        # Assuming you've ensured all attacks in your input files are covered, proceed with 5 classes
    
        y = np.array(y_a_classes)
        # One-Hot Encode (5 classes: 0, 1, 2, 3, 4)
        y = np_utils.to_categorical(y, num_classes=5)
    else:    
        #Converting to integers from the mappings file
        label_map = pd.read_csv(categoryMappingsPath + "41.csv", header=None)
        label_category = label_map.iloc[:, 0].values
        label_value = label_map.iloc[:, 1].values
        
        y = []
        for i in range(len(label_column)):
            y.append(label_value[label_category.tolist().index(label_column[i])])
        # Encoding the Dependent Variable
        y = np_utils.to_categorical(y)
    
   # Retrun the induvidual granular attack labels
    if return_subclass:
        #return X, y, subclass_column
        return X, y, label_column
    else:
        return X, y


