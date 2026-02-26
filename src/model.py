from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from typing import Union


class TaxiFareModel:
    """
    Модель для предсказания стоимости поездки на такси.
    Использует GradientBoostingRegressor из scikit-learn.
    """
    
    def __init__(self) -> None:
        """Инициализирует модель GradientBoostingRegressor."""
        self.model = GradientBoostingRegressor()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        Обучает модель на переданных данных.
        
        Args:
            X: Матрица признаков для обучения.
            y: Целевая переменная.
        """
        
        self.model.fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Выполняет предсказание для новых данных.
        
        Args:
            X: Матрица признаков для предсказания.
        
        Returns:
            np.ndarray: Массив предсказанных значений.
        """

        return self.model.predict(X)
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Вычисляет R^2 метрику на переданных данных.
        
        Args:
            X: Матрица признаков для оценки.
            y: Истинные значения целевой переменной.
        
        Returns:
            float: Значение R^2 метрики.
        """

        return self.model.score(X, y)