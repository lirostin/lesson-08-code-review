import pandas as pd


def add_time_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет временные признаки

    Args:
        df (pd.DataFrame): Получаемый датафрейм.

    Returns:
        pd.DataFrame: Датафрейм с добавленными признаками.
    """

    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
    return df

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Функция удяляет ненужные колонки и строки с пропущенными значениями.

    Args:
        df (pd.DataFrame): Получаемый датафрейм.

    Returns:
        pd.DataFrame: Датафрейм с удалёнными колонками и пропусками.
    """

    df = df.drop(['pickup_datetime', 'key'], axis=1, errors='ignore')
    df = df.dropna()
    return df