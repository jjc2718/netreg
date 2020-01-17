"""
Script to validate network regularization implementation on simulated
data.

"""
import os
import argparse
import subprocess
import logging
import tempfile
import pickle as pkl
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import config as cfg
import utilities.data_utilities as du
import simdata.simulate_loglinear as ll
import simdata.simulate_networks as snet
from utilities.classify_pytorch import TorchLR
from tcga_util import (
    train_model,
    get_threshold_metrics,
    extract_coefficients,
    align_matrices,
    check_status
)

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true',
               help='If flag is included, run PyTorch models on GPU')
p.add_argument('--num_features', type=int, default=10)
p.add_argument('--num_networks', type=int, default=2)
p.add_argument('--num_samples', type=int, default=100)
p.add_argument('--plot_learning_curves', default=None,
               help='If flag is included, plot learning curves and save\
                     them to this directory')
p.add_argument('--results_dir',
               default=cfg.repo_root.joinpath('pytorch_results').resolve(),
               help='Directory to write results to')
p.add_argument('--seed', type=int, default=cfg.default_seed)
p.add_argument('--uncorr_frac', type=float, default=0.5)
p.add_argument('--noise_stdev', type=float, default=0)
p.add_argument('--verbose', action='store_true')

prms = p.add_argument_group('pytorch_params')
prms.add_argument('--batch_size', type=int, default=None,
                  help='Batch size for PyTorch logistic regression')
prms.add_argument('--learning_rate', type=float, default=None,
                  help='Learning rate for PyTorch logistic regression')
prms.add_argument('--num_epochs', type=int, default=None,
                  help='Number of epochs for PyTorch logistic regression')
prms.add_argument('--l1_penalty', type=float, default=None,
                  help='L1 penalty multiplier for PyTorch logistic regression')
prms.add_argument('--param_search', action='store_true',
                  help='If flag is included, run a parameter search using the\
                        values in config.py and ignore provided parameters')

net = p.add_argument_group('netreg_params')
net.add_argument('--networks_dir', type=str,
                 default=cfg.repo_root.joinpath('simdata/sim_networks').resolve())
net.add_argument('--network_penalty', type=float, default=0,
                 help='Multiplier for network regularization term, defaults\
                       to 0 (no network penalty)')

args = p.parse_args()

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if (not args.param_search) and (None in [args.batch_size,
                                         args.learning_rate,
                                         args.num_epochs,
                                         args.l1_penalty]):
    import sys
    sys.exit('Error: must either include the "--param_search" flag'
             ' or manually pass a single value for all parameters')

if args.verbose:
    logger = logging.getLogger('netreg_benchmark')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

np.random.seed(args.seed)

if args.param_search:
    torch_params = cfg.netreg_param_choices
else:
    torch_params = {
        'batch_size': [args.batch_size],
        'l1_penalty': [args.l1_penalty],
        'learning_rate': [args.learning_rate],
        'num_epochs': [args.num_epochs],
        'network_penalty': [args.network_penalty]
    }

cv_results = {
    # TODO: could calculate R^2 for regression fit?
    'torch_train_mse': [],
    'torch_train_rmse': [],
    'torch_test_mse': [],
    'torch_test_rmse': [],
    'r_train_mse': [],
    'r_train_rmse': [],
    'r_test_mse': [],
    'r_test_rmse': [],
    'sklearn_train_mse': [],
    'sklearn_train_rmse': [],
    'sklearn_test_mse': [],
    'sklearn_test_rmse': []
}

# generate simulated data
train_frac = 0.8

X, betas, y, is_correlated, adj_matrix, network_groups = snet.simulate_network_reg(
        args.num_samples, args.num_features, args.uncorr_frac,
        args.num_networks, noise_stdev=args.noise_stdev, seed=args.seed,
        verbose=args.verbose)

train_ixs = ll.split_train_test(args.num_samples, train_frac, seed=args.seed,
                                verbose=True)
X_train, X_test = X[train_ixs], X[~train_ixs]
y_train, y_test = y[train_ixs], y[~train_ixs]

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

# make directory for learning curve if included
if args.plot_learning_curves is not None:
    learning_curves = True
    if not os.path.exists(args.plot_learning_curves):
        os.makedirs(args.plot_learning_curves)
else:
    learning_curves = False

# generate network for simulated data if it doesn't already exist
if not os.path.exists(args.networks_dir):
    os.makedirs(args.networks_dir)

network_filename = os.path.join(args.networks_dir,
                                'sim_groups_p{}_e{}_u{}_s{}.tsv'.format(
                                    args.num_features,
                                    args.noise_stdev,
                                    args.uncorr_frac,
                                    args.seed))

if not os.path.exists(network_filename):
    snet.save_numpy_to_el(adj_matrix, np.arange(args.num_samples),
                          network_filename)

###########################################################
# PYTORCH MODEL
###########################################################

logger.info('Running PyTorch model...')

# Fit PyTorch model on training set, test on held out data
torch_model = TorchLR(torch_params,
                      seed=args.seed,
                      network_file=network_filename,
                      network_features=np.ones(args.num_features).astype('bool'),
                      learning_curves=learning_curves,
                      use_gpu=args.gpu,
                      verbose=args.verbose)

losses, preds, preds_bn = torch_model.train_torch_model(X_train, X_test,
                                                        y_train, y_test,
                                                        save_weights=True)

torch_weights = torch_model.last_weights.flatten()
torch_pred_train, torch_pred_test = preds
torch_pred_bn_train, torch_pred_bn_test = preds_bn

# Calculate performance metrics
# TODO: this could be a function
torch_train_mse = mean_squared_error(y_train, torch_pred_train)
torch_train_rmse = np.sqrt(torch_train_mse)
torch_test_mse = mean_squared_error(y_test, torch_pred_test)
torch_test_rmse = np.sqrt(torch_test_mse)

cv_results['torch_train_mse'].append(torch_train_mse)
cv_results['torch_train_rmse'].append(torch_train_rmse)
cv_results['torch_test_mse'].append(torch_test_mse)
cv_results['torch_test_rmse'].append(torch_test_rmse)

best_torch_params = torch_model.best_params

# TODO: move to function somewhere, for learning curve functionality
# import matplotlib; matplotlib.use('Agg')
if args.plot_learning_curves is not None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    for metric in torch_model.monitor_.keys():
        num_epochs = len(torch_model.monitor_[metric])
        plt.plot(np.arange(1, num_epochs+1), torch_model.monitor_[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.savefig(os.path.join(args.plot_learning_curves,
                             'lc_n{}_p{}_e{}_u{}_s{}.pdf'.format(
                               args.num_samples, args.num_features,
                               args.noise_stdev, args.uncorr_frac,
                               args.seed)))

###########################################################
# R (netReg) MODEL
###########################################################

# Fit R model on training set an# test on held out data
r_args = [
    'Rscript',
    os.path.join(cfg.scripts_dir, 'run_netreg.R'),
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
    '--family', 'gaussian', # TODO change this for logistic regression
    '--l1_penalty', str(args.l1_penalty),
    '--network_penalty', str(args.network_penalty),
    '--num_epochs', str(args.num_epochs),
    '--learning_rate', str(args.learning_rate)
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
r_test_mse = mean_squared_error(y_test, r_pred_test)
r_test_rmse = np.sqrt(r_test_mse)

cv_results['r_train_mse'].append(r_train_mse)
cv_results['r_train_rmse'].append(r_train_rmse)
cv_results['r_test_mse'].append(r_test_mse)
cv_results['r_test_rmse'].append(r_test_rmse)

###########################################################
# SCIKIT-LEARN MODEL (BASELINE)
###########################################################

logger.info('Running scikit-learn SGD model (no network penalty)...')

from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(max_iter=args.num_epochs,
                   learning_rate='constant',
                   eta0=args.learning_rate,
                   penalty='l1',
                   alpha=args.l1_penalty)
reg.fit(X=X_train, y=y_train.flatten())
sklearn_pred_train = reg.predict(X_train)
sklearn_pred_test = reg.predict(X_test)

# Calculate performance metrics and extract model coefficients
sklearn_train_mse = mean_squared_error(y_train, sklearn_pred_train)
sklearn_train_rmse = np.sqrt(sklearn_train_mse)
sklearn_test_mse = mean_squared_error(y_test, sklearn_pred_test)
sklearn_test_rmse = np.sqrt(sklearn_test_mse)

s_coef = np.concatenate((reg.intercept_, reg.coef_))

cv_results['sklearn_train_mse'].append(sklearn_train_mse)
cv_results['sklearn_train_rmse'].append(sklearn_train_rmse)
cv_results['sklearn_test_mse'].append(sklearn_test_mse)
cv_results['sklearn_test_rmse'].append(sklearn_test_rmse)

# Save results to results directory
if hasattr(torch_model, 'results_df'):
    torch_model.results_df.to_csv(os.path.join(args.results_dir,
                                               'torch_params_{}.tsv'.format(
                                                   args.seed)), sep='\t')

cv_results_file = os.path.join(args.results_dir,
                               'cv_results_n{}_p{}_e{}_u{}_s{}.pkl'.format(
                                   args.num_samples, args.num_features,
                                   args.noise_stdev, args.uncorr_frac,
                                   args.seed))
true_coef_file = os.path.join(args.results_dir,
                              'true_coefs_n{}_p{}_e{}_u{}_s{}.txt'.format(
                                  args.num_samples, args.num_features,
                                  args.noise_stdev, args.uncorr_frac,
                                  args.seed))
torch_coef_file = os.path.join(args.results_dir,
                               'torch_coefs_n{}_p{}_e{}_u{}_s{}.txt'.format(
                                   args.num_samples, args.num_features,
                                   args.noise_stdev, args.uncorr_frac,
                                   args.seed))
sklearn_coef_file = os.path.join(args.results_dir,
                                 'sklearn_coefs_n{}_p{}_e{}_u{}_s{}.txt'.format(
                                   args.num_samples, args.num_features,
                                   args.noise_stdev, args.uncorr_frac,
                                   args.seed))

with open(cv_results_file, 'wb') as f:
    pkl.dump(cv_results, f)

np.savetxt(true_coef_file, betas, fmt='%.5f', delimiter='\t')
np.savetxt(torch_coef_file, torch_weights, fmt='%.5f', delimiter='\t')
np.savetxt(sklearn_coef_file, s_coef, fmt='%.5f', delimiter='\t')

