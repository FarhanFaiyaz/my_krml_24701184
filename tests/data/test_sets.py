import pytest
import pandas as pd

from my_krml_24701184.data.sets import pop_target


@pytest.fixture
def features_fixture():
    features_data = [
        [1, 25, "Junior"],
        [2, 33, "Confirmed"],
        [3, 42, "Manager"],
    ]
    return pd.DataFrame(features_data, columns=["employee_id", "age", "level"])

@pytest.fixture
def target_fixture():
    target_data = [5, 10, 20]
    return pd.Series(target_data, name="salary", copy=False)

def test_pop_target_with_data_fixture(features_fixture, target_fixture):
    input_df = features_fixture.copy()
    input_df["salary"] = target_fixture

    features, target = pop_target(df=input_df, target_col='salary')

    pd.testing.assert_frame_equal(features, features_fixture)
    pd.testing.assert_series_equal(target, target_fixture)

def test_pop_target_no_col_found(features_fixture, target_fixture):
    input_df = features_fixture.copy()

    with pytest.raises(KeyError):
        features, target = pop_target(df=input_df, target_col='salary')

def test_pop_target_col_none(features_fixture, target_fixture):
    input_df = features_fixture.copy()

    with pytest.raises(KeyError):
        features, target = pop_target(df=input_df, target_col=None)

def test_pop_target_df_none(features_fixture, target_fixture):
    input_df = features_fixture.copy()

    with pytest.raises(AttributeError):
        features, target = pop_target(df=None, target_col="salary")


#--------------------------------------------------------------------------------------------------------------------------------

import os
import pytest
import pandas as pd
from my_krml_24701184.data.sets import save_sets, load_sets

@pytest.fixture
def mock_data():
    return {
        'X_train': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
        'y_train': pd.DataFrame([1, 0]),
        'X_val': pd.DataFrame({'a': [5, 6], 'b': [7, 8]}),
        'y_val': pd.DataFrame([0, 1]),
        'X_test': pd.DataFrame({'a': [9, 10], 'b': [11, 12]}),
        'y_test': pd.DataFrame([1, 1])
    }

@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path

def test_save_sets(mock_data, temp_directory):
    save_sets(
        X_train=mock_data['X_train'],
        y_train=mock_data['y_train'],
        X_val=mock_data['X_val'],
        y_val=mock_data['y_val'],
        X_test=mock_data['X_test'],
        y_test=mock_data['y_test'],
        path=temp_directory
    )

    # Verify directory exists
    assert os.path.isdir(temp_directory)
    print(f"Temp directory: {temp_directory}")

    # Verify that files are created
    for filename in ['X_train.csv', 'y_train.csv', 'X_val.csv', 'y_val.csv', 'X_test.csv', 'y_test.csv']:
        file_path = temp_directory / filename
        assert os.path.isfile(file_path), f"File not found: {file_path}"
        print(f"File exists: {file_path}")

    # Verify the content of the files
    assert pd.read_csv(temp_directory / 'X_train.csv').equals(mock_data['X_train'])
    assert pd.read_csv(temp_directory / 'y_train.csv', header=None).equals(mock_data['y_train'])
    assert pd.read_csv(temp_directory / 'X_val.csv').equals(mock_data['X_val'])
    assert pd.read_csv(temp_directory / 'y_val.csv', header=None).equals(mock_data['y_val'])
    assert pd.read_csv(temp_directory / 'X_test.csv').equals(mock_data['X_test'])
    assert pd.read_csv(temp_directory / 'y_test.csv', header=None).equals(mock_data['y_test'])

def test_load_sets(mock_data, temp_directory):
    # Save mock data to temp_directory
    save_sets(
        X_train=mock_data['X_train'],
        y_train=mock_data['y_train'],
        X_val=mock_data['X_val'],
        y_val=mock_data['y_val'],
        X_test=mock_data['X_test'],
        y_test=mock_data['y_test'],
        path=temp_directory
    )

    loaded_data = load_sets(path=temp_directory)

    # Verify the loaded data
    assert 'X_train' in loaded_data
    assert 'y_train' in loaded_data
    assert 'X_val' in loaded_data
    assert 'y_val' in loaded_data
    assert 'X_test' in loaded_data
    assert 'y_test' in loaded_data

    assert loaded_data['X_train'].equals(mock_data['X_train'])
    assert loaded_data['y_train'].equals(mock_data['y_train'])
    assert loaded_data['X_val'].equals(mock_data['X_val'])
    assert loaded_data['y_val'].equals(mock_data['y_val'])
    assert loaded_data['X_test'].equals(mock_data['X_test'])
    assert loaded_data['y_test'].equals(mock_data['y_test'])

def test_load_sets_missing_files(mock_data, temp_directory):
    # Save only some mock data to temp_directory
    save_sets(
        X_train=mock_data['X_train'],
        y_train=mock_data['y_train'],
        path=temp_directory
    )

    loaded_data = load_sets(path=temp_directory)

    # Verify that only existing files are loaded
    assert 'X_train' in loaded_data
    assert 'y_train' in loaded_data
    assert 'X_val' not in loaded_data
    assert 'y_val' not in loaded_data
    assert 'X_test' not in loaded_data
    assert 'y_test' not in loaded_data



