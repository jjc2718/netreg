"""
Script to validate network regularization implementation on simulated data.

"""
import os
import time
import argparse
import subprocess
import logging
import tempfile
import pickle as pkl
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import config as cfg
import utilities.data_utilities as du
import simdata.simulate_loglinear as ll
import simdata.simulate_networks as snet
from tcga_util import (
    train_model,
    get_threshold_metrics,
    extract_coefficients,
    align_matrices,
    check_status
)

p = argparse.ArgumentParser()
p.add_argument('--add_frac', type=float, default=0.0)
p.add_argument('--add_only_uncorr', action='store_true')
p.add_argument('--gpu', action='store_true',
               help='If flag is included, run PyTorch models on GPU')
p.add_argument('--noise_stdev', type=float, default=0.0)
p.add_argument('--num_features', type=int, default=10)
p.add_argument('--num_networks', type=int, default=2)
p.add_argument('--num_samples', type=int, default=100)
p.add_argument('--plot_learning_curves', default=None,
               help='If flag is included, plot learning curves and save\
                     them to this directory')
p.add_argument('--remove_frac', type=float, default=0.0)
p.add_argument('--results_dir',
               default=cfg.repo_root.joinpath('pytorch_results').resolve(),
               help='Directory to write results to')
p.add_argument('--seed', type=int, default=cfg.default_seed)
p.add_argument('--uncorr_frac', type=float, default=0.5)
p.add_argument('--verbose', action='store_true')

prms = p.add_argument_group('model_params')
prms.add_argument('--l1_penalty', type=float, default=0.1,
                  help='L1 penalty multiplier for PyTorch logistic regression')
prms.add_argument('--num_epochs', type=int, default=100,
                  help='Number of epochs for PyTorch logistic regression')
prms.add_argument('--networks_dir', type=str,
                 default=cfg.repo_root.joinpath('simdata/sim_networks').resolve())
prms.add_argument('--network_penalty', type=float, default=0,
                 help='Multiplier for network regularization term, defaults\
                       to 0 (no network penalty)')
prms.add_argument('--ignore_network', action='store_true')

args = p.parse_args()

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.verbose:
    logger = logging.getLogger('gelnet_benchmark')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

np.random.seed(args.seed)

model_params = {
    'l1_penalty': [args.l1_penalty],
    'num_epochs': [args.num_epochs],
    'network_penalty': [args.network_penalty]
}

# if ignore_network flag included, run pytorch with no network penalty
# this is useful as a baseline
if args.ignore_network:
    torch_params['network_penalty'] = [0.0]

cv_results = {
    'r_train_rmse': [],
    'r_test_rmse': [],
    'r_train_r2': [],
    'r_test_r2': [],
    'sklearn_train_rmse': [],
    'sklearn_train_r2': [],
    'sklearn_test_rmse': [],
    'sklearn_test_r2': []
}

# generate simulated data
X, betas, y, is_correlated, adj_matrix, network_groups = snet.simulate_network_reg(
        args.num_samples, args.num_features, args.uncorr_frac,
        args.num_networks, noise_stdev=args.noise_stdev, seed=args.seed,
        add_frac=args.add_frac, remove_frac=args.remove_frac,
        add_only_uncorr=args.add_only_uncorr, verbose=args.verbose)

# split simulated data into train/test sets (and optionally tune set)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=args.seed)

if args.verbose:
    logger.info('Train samples: {}, test samples: {}'.format(
        X_train.shape[0], X_test.shape[0]))

# make directory for learning curve if included
if args.plot_learning_curves is not None:
    learning_curves = True
    if not os.path.exists(args.plot_learning_curves):
        os.makedirs(args.plot_learning_curves)
else:
    learning_curves = False

# generate network for simulated data if it doesn't already exist
# this only has to be done once, even if we're doing a parameter search
if not os.path.exists(args.networks_dir):
    os.makedirs(args.networks_dir)

network_filename = os.path.join(args.networks_dir,
                                'sim_groups_p{}_u{}_a{}_r{}_s{}.tsv'.format(
                                    args.num_features,
                                    args.uncorr_frac,
                                    args.add_frac,
                                    args.remove_frac,
                                    args.seed))

if not os.path.exists(network_filename):
    snet.save_numpy_to_el(adj_matrix, np.arange(args.num_samples),
                          network_filename)

###########################################################
# R (netReg) MODEL
###########################################################

# generate tempfiles for train/test data, to pass to R script
train_data = tempfile.NamedTemporaryFile(mode='w', delete=False)
test_data = tempfile.NamedTemporaryFile(mode='w', delete=False)
train_labels = tempfile.NamedTemporaryFile(mode='w', delete=False)
test_labels = tempfile.NamedTemporaryFile(mode='w', delete=False)
np.savetxt(train_data, X_train, fmt='%.5f', delimiter='\t')
np.savetxt(test_data, X_test, fmt='%.5f', delimiter='\t')
np.savetxt(train_labels, y_train, fmt='%i')
np.savetxt(test_labels, y_test, fmt='%i')
Filenames = namedtuple('Filenames', ['train_data', 'test_data',
                                     'train_labels', 'test_labels'])
fnames = Filenames(
    train_data=train_data.name,
    test_data=test_data.name,
    train_labels=train_labels.name,
    test_labels=test_labels.name
)
train_data.close()
test_data.close()
train_labels.close()
test_labels.close()

# Fit R model on training set an# test on held out data
r_args = [
    'Rscript',
    os.path.join(cfg.scripts_dir, 'run_gelnet.R'),
    '--train_data', fnames.train_data,
    '--test_data', fnames.test_data,
    '--train_labels', fnames.train_labels,
    '--test_labels', fnames.test_labels,
    '--network_file', network_filename,
    '--num_samples', str(args.num_samples),
    '--num_features', str(args.num_features),
    '--noise_stdev', str(args.noise_stdev),
    '--uncorr_frac', str(args.uncorr_frac),
    '--results_dir', args.results_dir,
    '--seed', str(args.seed),
    '--l1_penalty', str(args.l1_penalty),
    '--network_penalty', str(args.network_penalty),
    '--num_epochs', str(args.num_epochs)
]
if args.verbose:
    r_args.append('--verbose')
    logger.info('Running: {}'.format(' '.join(r_args)))

r_env = os.environ.copy()
r_env['MKL_THREADING_LAYER'] = 'GNU'
subprocess.check_call(r_args, env=r_env)

# clean up temp files
for fname in fnames:
    os.remove(fname)

r_pred_train = np.loadtxt(
    os.path.join(args.results_dir,
                 'r_preds_train_n{}_p{}_e{}_u{}_s{}.txt'.format(
                     args.num_samples, args.num_features,
                     args.noise_stdev, args.uncorr_frac,
                     args.seed)),
    delimiter='\t')
r_pred_test = np.loadtxt(
    os.path.join(args.results_dir,
                 'r_preds_test_n{}_p{}_e{}_u{}_s{}.txt'.format(
                     args.num_samples, args.num_features,
                     args.noise_stdev, args.uncorr_frac,
                     args.seed)),
    delimiter='\t')

# get binary predictions
r_pred_bn_train = (r_pred_train > 0.5).astype('int')
r_pred_bn_test = (r_pred_test > 0.5).astype('int')

# Calculate performance metrics
r_train_mse = mean_squared_error(y_train, r_pred_train)
r_train_rmse = np.sqrt(r_train_mse)
r_train_r2 = r2_score(y_train, r_pred_train)
r_test_mse = mean_squared_error(y_test, r_pred_test)
r_test_rmse = np.sqrt(r_test_mse)
r_test_r2 = r2_score(y_test, r_pred_test)

cv_results['r_train_rmse'].append(r_train_rmse)
cv_results['r_train_r2'].append(r_train_r2)
cv_results['r_test_rmse'].append(r_test_rmse)
cv_results['r_test_r2'].append(r_test_r2)

###########################################################
# SCIKIT-LEARN MODEL (BASELINE)
###########################################################

if args.verbose:
    logger.info('##### Running parameter search for scikit-learn model... #####')

from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(max_iter=model_params['num_epochs'][0],
                   learning_rate='constant',
                   eta0=0.01,
                   penalty='l1',
                   alpha=model_params['l1_penalty'][0])
reg.fit(X=X_train, y=y_train.flatten())
sklearn_pred_train = reg.predict(X_train)
sklearn_pred_test = reg.predict(X_test)

# Calculate performance metrics and extract model coefficients
sklearn_train_mse = mean_squared_error(y_train, sklearn_pred_train)
sklearn_train_rmse = np.sqrt(sklearn_train_mse)
sklearn_train_r2 = r2_score(y_train, sklearn_pred_train)
sklearn_test_mse = mean_squared_error(y_test, sklearn_pred_test)
sklearn_test_rmse = np.sqrt(sklearn_test_mse)
sklearn_test_r2 = r2_score(y_test, sklearn_pred_test)

s_coef = np.concatenate((reg.intercept_, reg.coef_))

cv_results['sklearn_train_rmse'].append(sklearn_train_rmse)
cv_results['sklearn_train_r2'].append(sklearn_train_r2)
cv_results['sklearn_test_rmse'].append(sklearn_test_rmse)
cv_results['sklearn_test_r2'].append(sklearn_test_r2)

# Save results to results directory
cv_results_file = os.path.join(args.results_dir,
                               'cv_results_n{}_p{}_u{}_a{}_r{}_s{}.pkl'.format(
                                   args.num_samples, args.num_features,
                                   args.uncorr_frac, args.add_frac,
                                   args.remove_frac, args.seed))
true_coef_file = os.path.join(args.results_dir,
                              'true_coefs_n{}_p{}_u{}_a{}_r{}_s{}.txt'.format(
                                   args.num_samples, args.num_features,
                                   args.uncorr_frac, args.add_frac,
                                   args.remove_frac, args.seed))
sklearn_coef_file = os.path.join(args.results_dir,
                                 'sklearn_coefs_n{}_p{}_u{}_a{}_r{}_s{}.txt'.format(
                                   args.num_samples, args.num_features,
                                   args.uncorr_frac, args.add_frac,
                                   args.remove_frac, args.seed))

with open(cv_results_file, 'wb') as f:
    pkl.dump(cv_results, f)

np.savetxt(true_coef_file, betas, fmt='%.5f', delimiter='\t')
np.savetxt(sklearn_coef_file, s_coef, fmt='%.5f', delimiter='\t')

print(cv_results)

