# Предсказание стоимости поездки в сервисе такси

Модель для предсказания стоимости поездки на такси на основе Gradient Boosting.

## Зависимости

- Python 3.8+
- pandas
- scikit-learn
- pytest (для тестов)


## Установка

1. Клонируйте репозиторий

```
git clone git@github.com:lirostin/lesson-08-code-review.git
```

2. Создайте виртуальное окржуение

```
make install_python_macos
make install_uv_mac_linux
make init_uv
```

3. Запуск проекта 

```
make run_project
```

### Структура проекта 

taxi-fare-prediction/
├── data/
│   └── uber.csv           # исходные данные
├── src/
│   ├── data.py            # загрузка и разделение данных
│   ├── features.py        # обработка признаков
│   └── model.py           # определение модели
├── tests/
│   └── test_data.py       # тесты для загрузки данных
├── main.py                 # основной скрипт
└── README.md