import pandas as pd
from datetime import datetime


def add_time_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
    # Удаляем ненужные колонки
    df = df.drop(['pickup_datetime', 'key'], axis=1)
    
    # Удаляем строки с пропущенными значениями
    df = df.dropna()
    
    return df