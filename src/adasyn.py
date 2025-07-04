
# Load this training data (80% fraud + 80% normal samples).
# Uses ADASYN to create synthetic fraud transactions, so your dataset is balanced.
# Saves these new balanced datasets into .pkl files so you can quickly load them for ML training later.

from imblearn.over_sampling import ADASYN
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle

import os

# Create the directory if it doesn't exist
os.makedirs('pickle', exist_ok=True)


# Load data
data = pd.read_csv("C:/Users/Lenovo/Desktop/credit_card_fraud_detection/data/creditcard.csv")
fraud = shuffle(data[data.Class == 1])
normal = shuffle(data[data.Class == 0])

# Sample
X_train = pd.concat([
    fraud.sample(frac=0.8),
    normal.sample(frac=0.8)
], axis=0)

# Prepare X, y
X = X_train.drop(columns=['Class']).values
y = X_train['Class'].values

# Run ADASYN
ada = ADASYN()
X_resampled, y_resampled = ada.fit_resample(X, y)

# Save
with open('pickle/train_data_resampled.pkl', 'wb') as f:
    pickle.dump(X_resampled, f)
with open('pickle/train_data_labels_resampled.pkl', 'wb') as f:
    pickle.dump(y_resampled, f)


# Converting the resampled trained data in to dataframe

