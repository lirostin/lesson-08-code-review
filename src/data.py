import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """
    Загружает данные из CSV файла.
    
    Args:
        path (str): Путь к CSV файлу.
    
    Returns:
        pd.DataFrame: Загруженные данные.
    
    Raises:
        FileNotFoundError: Если файл не найден по указанному пути.
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path)

def split_data(
    data: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет данные на обучающую и тестовую выборки.
    
    Args:
        data (pd.DataFrame): Исходные данные.
        test_size (float): Доля тестовой выборки (по умолчанию 0.2).
        random_state (int): Seed для воспроизводимости (по умолчанию 42).
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test.
    """

    features = data.drop('fare_amount', axis=1)
    target = data['fare_amount']
    return train_test_split(features, target, test_size=test_size, random_state=random_state)
