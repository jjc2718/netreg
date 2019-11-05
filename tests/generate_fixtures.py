import os
import numpy as np
import pandas as pd

import sys; sys.path.append('.')
import config as cfg
import simdata.simulate_loglinear as ll
from tcga_util import train_model, extract_coefficients

def generate_data(seed):
    np.random.seed(seed)
    params = {
        'n_train': 10,
        'n_test': 10,
        'p': 10,
    }

    # generate simulated data from a log-linear model
    # the details of the data don't really matter here, we're just making
    # sure the model gives the same output as before
    X, y, _, __ = ll.simulate_ll(params['n_train']+params['n_test'],
                                 params['p'], 0, seed=seed)
    train_ixs = ll.split_train_test(params['n_train']+params['n_test'],
                                    params['n_train']/(params['n_train']+params['n_test']),
                                    seed=seed)
    X_train, X_test = X[train_ixs], X[~train_ixs]
    y_train, y_test = y[train_ixs], y[~train_ixs]

    # put things into dataframes
    train_index = ['S{}'.format(i) for i in range(params['n_train'])]
    test_index = ['S{}'.format(i) for i in range(
                   params['n_train'], params['n_train']+params['n_test'])]
    columns = ['G{}'.format(j) for j in range(params['p'])]
    X_train_df = pd.DataFrame(X_train, index=train_index, columns=columns)
    X_test_df = pd.DataFrame(X_test, index=test_index, columns=columns)
    y_train_df = pd.DataFrame(y_train, index=train_index, columns=['status'])
    y_test_df = pd.DataFrame(y_test, index=train_index, columns=['status'])

    return X_train_df, X_test_df, y_train_df, y_test_df


def predict(X_train, X_test, y_train, y_test):

    cv_pipeline, y_pred_train, y_pred_test, y_cv = train_model(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        alphas=cfg.alphas,
        l1_ratios=cfg.l1_ratios,
        n_folds=cfg.folds,
        max_iter=cfg.max_iter
    )

    y_pred_train_df = pd.DataFrame(y_pred_train,
                                   index=y_train.index,
                                   columns=y_train.columns)
    y_pred_test_df = pd.DataFrame(y_pred_test,
                                  index=y_test.index,
                                  columns=y_test.columns)

    coef_df = extract_coefficients(
        cv_pipeline=cv_pipeline,
        feature_names=X_train.columns,
        signal='signal',
        z_dim=cfg.num_features_raw,
        seed=cfg.default_seed,
        algorithm='raw'
    )

    return y_pred_train_df, y_pred_test_df, coef_df


if __name__ == '__main__':

    # generate data
    X_train_df, X_test_df, y_train_df, y_test_df = generate_data(cfg.default_seed)

    # make predictions
    y_pred_train_df, y_pred_test_df, coef_df = predict(
        X_train_df, X_test_df, y_train_df, y_test_df)

    # save results
    if not os.path.exists(cfg.fixtures_dir):
        os.makedirs(cfg.fixtures_dir)

    y_pred_train_df.to_csv(
        cfg.saved_results_train, sep="\t", index=False,
        compression="gzip", float_format="%.5g"
    )
    y_pred_test_df.to_csv(
        cfg.saved_results_test, sep="\t", index=False,
        compression="gzip", float_format="%.5g"
    )
    coef_df.to_csv(
        cfg.saved_coefs, sep="\t", index=False,
        compression="gzip", float_format="%.5g"
    )

