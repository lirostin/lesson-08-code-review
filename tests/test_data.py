from src.data import load_data
from main import get_csv_path

def test_load_data():
    df = load_data(get_csv_path())
    assert not df.empty
    assert 'fare_amount' in df.columns
