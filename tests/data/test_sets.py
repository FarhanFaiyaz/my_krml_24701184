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
import pandas as pd
import pytest

from my_krml_24701184.data.sets import save_sets  

@pytest.fixture
def tmp_save_path(tmp_path):
    return str(tmp_path)

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0], name="target")
    X_val = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})
    y_val = pd.Series([1, 0], name="target")
    X_test = pd.DataFrame({"feature1": [11, 12], "feature2": [13, 14]})
    y_test = pd.Series([1, 1], name="target")
    return X_train, y_train, X_val, y_val, X_test, y_test

def test_save_sets_all(tmp_save_path, sample_data):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    # Call the function
    save_sets(
        base_path=tmp_save_path, 
        X_train=X_train, 
        y_train=y_train, 
        X_val=X_val, 
        y_val=y_val, 
        X_test=X_test, 
        y_test=y_test
    )

    # Check that all files exist
    assert os.path.exists(os.path.join(tmp_save_path, "X_train.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "y_train.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "X_val.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "y_val.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "X_test.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "y_test.csv"))

    # Check the content of one file
    saved_X_train = pd.read_csv(os.path.join(tmp_save_path, "X_train.csv"))
    pd.testing.assert_frame_equal(saved_X_train, X_train)

def test_save_sets_partial(tmp_save_path, sample_data):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    # Call the function with only training and validation data
    save_sets(
        base_path=tmp_save_path, 
        X_train=X_train, 
        y_train=y_train, 
        X_val=X_val, 
        y_val=y_val
    )

    # Check that only the specified files exist
    assert os.path.exists(os.path.join(tmp_save_path, "X_train.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "y_train.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "X_val.csv"))
    assert os.path.exists(os.path.join(tmp_save_path, "y_val.csv"))
    assert not os.path.exists(os.path.join(tmp_save_path, "X_test.csv"))
    assert not os.path.exists(os.path.join(tmp_save_path, "y_test.csv"))

def test_save_sets_empty(tmp_save_path):
    # Call the function with no data
    save_sets(base_path=tmp_save_path)

    # Check that no files were created
    assert not os.listdir(tmp_save_path)

def test_save_sets_invalid_path(sample_data):
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data

    # Try saving to an invalid path and expect an OSError
    with pytest.raises(OSError):
        save_sets(
            base_path="/invalid/path", 
            X_train=X_train, 
            y_train=y_train
        )


#--------------------------------------------------------------------------------------------------------------------------------

# test_load_sets.py
import os
import pandas as pd
import pytest

from my_krml_24701184.data.sets import load_sets 

@pytest.fixture
def setup_csv_files(tmp_path):
    # Create sample CSV files
    data = {
        "X_train": pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]}),
        "y_train": pd.Series([0, 1], name="target"),
        "X_val": pd.DataFrame({"feature1": [5, 6], "feature2": [7, 8]}),
        "y_val": pd.Series([1, 0], name="target"),
        "X_test": pd.DataFrame({"feature1": [9, 10], "feature2": [11, 12]}),
        "y_test": pd.Series([0, 1], name="target"),
    }

    for name, df in data.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)

    return tmp_path

def test_load_sets_all(setup_csv_files):
    base_path = str(setup_csv_files)
    datasets = load_sets(base_path)

    assert "X_train" in datasets
    assert "y_train" in datasets
    assert "X_val" in datasets
    assert "y_val" in datasets
    assert "X_test" in datasets
    assert "y_test" in datasets

    # Check data
    pd.testing.assert_frame_equal(datasets["X_train"], pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]}))
    pd.testing.assert_series_equal(datasets["y_train"], pd.Series([0, 1], name="target"))

def test_load_sets_partial(setup_csv_files):
    # Delete some files
    os.remove(setup_csv_files / "X_test.csv")
    os.remove(setup_csv_files / "y_test.csv")

    base_path = str(setup_csv_files)
    datasets = load_sets(base_path)

    assert "X_train" in datasets
    assert "y_train" in datasets
    assert "X_val" in datasets
    assert "y_val" in datasets
    assert "X_test" not in datasets
    assert "y_test" not in datasets

def test_load_sets_empty(tmp_path):
    base_path = str(tmp_path)
    datasets = load_sets(base_path)

    assert not datasets  # No files should be loaded

