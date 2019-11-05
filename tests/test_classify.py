"""
Regression tests for logistic regression/classification implementation

"""
import os
import pytest
import numpy as np
import pandas as pd

import generate_fixtures as fx

import sys; sys.path.append('.')
import config as cfg
import simdata.simulate_loglinear as ll

@pytest.fixture
def classify_data():
    X_train_df, X_test_df, y_train_df, y_test_df = fx.generate_data(cfg.default_seed)
    return X_train_df, X_test_df, y_train_df, y_test_df


@pytest.fixture
def saved_results():
    try:
        y_pred_train_df = pd.read_csv(cfg.saved_results_train, sep='\t')
        y_pred_test_df = pd.read_csv(cfg.saved_results_test, sep='\t')
        coef_df = pd.read_csv(cfg.saved_coefs, sep='\t')
    except OSError:
        y_pred_train_df = None
        y_pred_test_df = None
        coef_df = None
    return y_pred_train_df, y_pred_test_df, coef_df


def test_lr(classify_data, saved_results):
    """Compare logistic regression output to saved results."""

    X_train_df, X_test_df, y_train_df, y_test_df = classify_data
    saved_preds_train, saved_preds_test, saved_coefs = saved_results

    assert (saved_preds_train is not None and
            saved_preds_test is not None and
            saved_coefs is not None), (
            'Saved fixtures not found. Please run generate_fixtures.py')

    y_pred_train_df, y_pred_test_df, coef_df = fx.predict(
            X_train_df, X_test_df, y_train_df, y_test_df)

    assert np.allclose(y_pred_train_df['status'].values,
                       saved_preds_train['status'].values,
                       atol=1e-4)
    assert np.allclose(y_pred_test_df['status'].values,
                       saved_preds_test['status'].values,
                       atol=1e-4)
    assert np.allclose(np.sort(coef_df['weight'].values),
                       np.sort(saved_coefs['weight'].values),
                       atol=1e-4)

