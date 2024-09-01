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



import pytest
import numpy as np
import os
from my_krml_24701184.data.sets import save_sets, load_sets

@pytest.fixture
def temp_data():
    """Fixture for creating temporary data and directory."""
    temp_path = './temp_data/'
    os.makedirs(temp_path, exist_ok=True)

    # Sample data
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    X_val = np.array([[5, 6]])
    y_val = np.array([1])
    X_test = np.array([[7, 8]])
    y_test = np.array([0])

    # Save the sample data
    save_sets(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, path=temp_path)

    yield temp_path

    # Cleanup after tests
    for file in os.listdir(temp_path):
        os.remove(os.path.join(temp_path, file))
    os.rmdir(temp_path)

def test_save_sets(temp_data):
    """Test the save_sets function."""
    path = temp_data

    assert os.path.isfile(f'{path}X_train.npy')
    assert os.path.isfile(f'{path}y_train.npy')
    assert os.path.isfile(f'{path}X_val.npy')
    assert os.path.isfile(f'{path}y_val.npy')
    assert os.path.isfile(f'{path}X_test.npy')
    assert os.path.isfile(f'{path}y_test.npy')

def test_load_sets(temp_data):
    """Test the load_sets function."""
    path = temp_data

    X_train, y_train, X_val, y_val, X_test, y_test = load_sets(path=path)

    expected_X_train = np.array([[1, 2], [3, 4]])
    expected_y_train = np.array([0, 1])
    expected_X_val = np.array([[5, 6]])
    expected_y_val = np.array([1])
    expected_X_test = np.array([[7, 8]])
    expected_y_test = np.array([0])

    np.testing.assert_array_equal(X_train, expected_X_train)
    np.testing.assert_array_equal(y_train, expected_y_train)
    np.testing.assert_array_equal(X_val, expected_X_val)
    np.testing.assert_array_equal(y_val, expected_y_val)
    np.testing.assert_array_equal(X_test, expected_X_test)
    np.testing.assert_array_equal(y_test, expected_y_test)