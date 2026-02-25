import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_data(data, test_size=0.2, random_state=42):
    features = data.drop('fare_amount', axis=1)
    target = data['fare_amount']
    return train_test_split(features, target, test_size=test_size, random_state=random_state)
