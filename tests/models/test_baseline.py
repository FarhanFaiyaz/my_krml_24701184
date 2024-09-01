import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from my_krml_24701184.models.baseline import BaselineModel 

@pytest.fixture
def y_train_fixture():
    return pd.Series([0, 0, 1, 0, 1, 0, 0])

@pytest.fixture
def y_val_fixture():
    return pd.Series([0, 1, 0, 1])

@pytest.fixture
def X_val_fixture():
    return pd.DataFrame({
        'feature1': [10, 20, 30, 40],
        'feature2': [1, 2, 3, 4]
    })

def test_baseline_model_initialization(y_train_fixture, X_val_fixture, y_val_fixture):
    baseline_model = BaselineModel(y_train=y_train_fixture, X_val=X_val_fixture, y_val=y_val_fixture)
    assert baseline_model.y_mode == 0, "The mode should be 0."

def test_compute_baseline_predictions(y_train_fixture, X_val_fixture, y_val_fixture):
    baseline_model = BaselineModel(y_train=y_train_fixture, X_val=X_val_fixture, y_val=y_val_fixture)
    y_base = baseline_model.compute_baseline_predictions()
    expected = np.array([0, 0, 0, 0])
    np.testing.assert_array_equal(y_base, expected, "Baseline predictions do not match expected values.")

def test_calculate_auroc(y_train_fixture, X_val_fixture, y_val_fixture):
    baseline_model = BaselineModel(y_train=y_train_fixture, X_val=X_val_fixture, y_val=y_val_fixture)
    auroc = baseline_model.calculate_auroc()
    
    # Manually calculate the expected AUROC for comparison
    y_base_proba = np.full(len(y_val_fixture), 0.0)
    expected_auroc = roc_auc_score(y_val_fixture, y_base_proba)
    
    assert np.isclose(auroc, expected_auroc, atol=1e-6), "AUROC calculation does not match expected value."

def test_plot_roc_curve(y_train_fixture, X_val_fixture, y_val_fixture):
    baseline_model = BaselineModel(y_train=y_train_fixture, X_val=X_val_fixture, y_val=y_val_fixture)
    
    # This test checks if the plot method runs without errors
    baseline_model.plot_roc_curve()
    

def test_evaluate(y_train_fixture, X_val_fixture, y_val_fixture):
    baseline_model = BaselineModel(y_train=y_train_fixture, X_val=X_val_fixture, y_val=y_val_fixture)
    auroc = baseline_model.evaluate()
    
    # Manually calculate the expected AUROC for comparison
    y_base_proba = np.full(len(y_val_fixture), 0.0)
    expected_auroc = roc_auc_score(y_val_fixture, y_base_proba)
    
    assert np.isclose(auroc, expected_auroc, atol=1e-6), "Evaluate method's AUROC does not match expected value."
