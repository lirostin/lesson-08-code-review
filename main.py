from src.data import load_data, split_data
from src.features import add_time_features, clean_data
from src.model import TaxiFareModel
import argparse


def get_csv_path() -> str:
    """
    Возращаяет путь к файлу с данными.

    Returns:
        str: Путь к файлу в виде строки.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="data/uber.csv")
    args = parser.parse_args()
    return args.csv_path


if __name__ == "__main__":
    # Загрузка и обработка данных
    raw_data = load_data(get_csv_path())
    processed_data = add_time_features(raw_data)
    clear_data = clean_data(processed_data)
    X_train, X_test, y_train, y_test = split_data(clear_data)

    # Обучение модели
    model = TaxiFareModel()
    model.fit(X_train, y_train)

    # Оценка
    score = model.score(X_test, y_test)
    print(f"R²: {score:.2f}")
