# Here I splitted the data into training, testing and validation set
import pandas as pd
from collections import Counter
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pandas as pd



def data_split_function(data_path="C:/Users/Lenovo/Desktop/credit_card_fraud_detection/data/creditcard.csv"):
    data = pd.read_csv(data_path)
    fraud = shuffle(data[data.Class == 1])
    normal = shuffle(data[data.Class == 0])

    # Produce a training set of 80% of fraudulent and 80% of normal transaction
    X_train = fraud.sample(frac=0.8)
    X_train = pd.concat([X_train, normal.sample(frac=0.8)], axis=0)

    # Split remainder into testing and validation
    remainder = data.loc[~data.index.isin(X_train.index)]
    X_test = remainder.sample(frac=0.7)
    X_validation = remainder.loc[~remainder.index.isin(X_test.index)]

    return X_train, X_test, X_validation



def scale_with_standard_scaler(X_train, X_test, X_validation, data_full):
    """
    Scale features to mean=0, std=1 using StandardScaler fit on data_full (excluding 'Class').
    Keeps 'Class' column unchanged.

    Returns new DataFrames with scaled features + original 'Class'.
    """
    scaler = StandardScaler()
    scaler.fit(data_full.drop('Class', axis=1))

    def scale_split(df):
        scaled_features = scaler.transform(df.drop('Class', axis=1))
        df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1], index=df.index)
        df_scaled['Class'] = df['Class']
        return df_scaled

    X_train_scaled = scale_split(X_train)
    X_test_scaled = scale_split(X_test)
    X_validation_scaled = scale_split(X_validation)

    return X_train_scaled, X_test_scaled, X_validation_scaled

if __name__ == "__main__":
    print("✅ Functions loaded successfully.")