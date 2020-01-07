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

import config as cfg
import utilities.data_utilities as du
import simdata.simulate_loglinear as ll
from utilities.classify_pytorch import TorchLR
from tcga_util import get_threshold_metrics

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true',
               help='If flag is included, run PyTorch models on GPU')
p.add_argument('--results_dir',
               default=cfg.repo_root.joinpath('pytorch_results').resolve(),
               help='where to write results to')
p.add_argument('--seed', type=int, default=cfg.default_seed)
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
net.add_argument('--network_file', type=str, default=None,
                 help='Path to network file (in edge list format), if\
                       not included no network regularization is used')
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
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

np.random.seed(args.seed)

if args.param_search and (args.network_file is not None):
    torch_params = cfg.netreg_param_choices
elif args.param_search:
    torch_params = cfg.torch_param_choices
else:
    torch_params = {
        'batch_size': [args.batch_size],
        'l1_penalty': [args.l1_penalty],
        'learning_rate': [args.learning_rate],
        'num_epochs': [args.num_epochs]
    }
    if args.network_file is not None:
        torch_params['network_penalty'] = [args.network_penalty]

cv_results = {
    'r_train_auroc': [],
    'r_train_aupr': [],
    'r_train_acc': [],
    'r_test_auroc': [],
    'r_test_aupr': [],
    'r_test_acc': [],
    'torch_train_auroc': [],
    'torch_train_aupr': [],
    'torch_train_acc': [],
    'torch_test_auroc': [],
    'torch_test_aupr': [],
    'torch_test_acc': []
}

# generate simulated data
# TODO: these should be argparse arguments
n = 100
p = 10
uncorr_frac = 0.5
train_frac = 0.8
X, y, _, is_correlated = ll.simulate_ll(n, p, uncorr_frac, seed=args.seed,
                                        verbose=args.verbose, unit_coefs=True)
train_ixs = ll.split_train_test(n, train_frac, seed=args.seed, verbose=True)
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
# TODO: make these a named tuple
fnames = [
    train_data.name,
    test_data.name,
    train_labels.name,
    test_labels.name
]
train_data.close()
test_data.close()
train_labels.close()
test_labels.close()

# Fit torch model on training set, test on held out data
torch_model = TorchLR(torch_params,
                      seed=args.seed,
                      network_file=args.network_file,
                      use_gpu=args.gpu,
                      verbose=args.verbose,
                      network_features=np.ones(p).astype('bool'),
                      correlated_features=is_correlated)

losses, preds, preds_bn = torch_model.train_torch_model(X_train, X_test,
                                                        y_train, y_test,
                                                        save_weights=True)

torch_weights = torch_model.last_weights.flatten()
torch_pred_train, torch_pred_test = preds
torch_pred_bn_train, torch_pred_bn_test = preds_bn
# TODO: this can be a function
torch_train_acc = TorchLR.calculate_accuracy(y_train,
                                             torch_pred_bn_train.flatten())
torch_test_acc = TorchLR.calculate_accuracy(y_test,
                                            torch_pred_bn_test.flatten())
torch_train_results = get_threshold_metrics(
        y_train, torch_pred_train, drop=False
)
torch_test_results = get_threshold_metrics(
    y_test, torch_pred_test, drop=False
)
cv_results['torch_train_auroc'].append(torch_train_results['auroc'])
cv_results['torch_train_aupr'].append(torch_train_results['aupr'])
cv_results['torch_test_auroc'].append(torch_test_results['auroc'])
cv_results['torch_test_aupr'].append(torch_test_results['aupr'])
cv_results['torch_train_acc'].append(
        TorchLR.calculate_accuracy(y_train,
                                   torch_pred_bn_train.flatten()))
cv_results['torch_test_acc'].append(
        TorchLR.calculate_accuracy(y_test,
                                   torch_pred_bn_test.flatten()))
best_torch_params = torch_model.best_params

# Fit R netReg (TensorFlow) model on training set and
# test on held out data
r_args = [
    'Rscript',
    os.path.join(cfg.scripts_dir, 'run_netreg.R'),
    '--train_data', fnames[0],
    '--test_data', fnames[1],
    '--train_labels', fnames[2],
    '--test_labels', fnames[3],
    '--network_file', args.network_file,
    '--results_dir', args.results_dir,
    '--seed', str(args.seed),
    '--l1_penalty', str(args.l1_penalty),
    '--network_penalty', str(args.network_penalty),
    '--num_epochs', str(args.num_epochs),
    '--learning_rate', str(args.learning_rate)
]
if args.verbose:
    r_args.append('--verbose')

logging.info('Running: {}'.format(' '.join(r_args)))

r_env = os.environ.copy()
r_env['MKL_THREADING_LAYER'] = 'GNU'
subprocess.check_call(r_args, env=r_env)

# clean up temp files
for fname in fnames:
    os.remove(fname)

r_pred_train = np.loadtxt(
    os.path.join(args.results_dir,
                 'r_preds_train_{}.tsv'.format(args.seed)),
    delimiter='\t')
r_pred_test = np.loadtxt(
    os.path.join(args.results_dir,
                 'r_preds_test_{}.tsv'.format(args.seed)),
    delimiter='\t')

# get binary predictions
r_pred_bn_train = (r_pred_train > 0.5).astype('int')
r_pred_bn_test = (r_pred_test > 0.5).astype('int')

r_train_results = get_threshold_metrics(
    y_train, r_pred_train, drop=False
)
r_test_results = get_threshold_metrics(
    y_test, r_pred_test, drop=False
)

cv_results['r_train_auroc'].append(r_train_results['auroc'])
cv_results['r_train_aupr'].append(r_train_results['aupr'])
cv_results['r_test_auroc'].append(r_test_results['auroc'])
cv_results['r_test_aupr'].append(r_test_results['aupr'])
cv_results['r_train_acc'].append(
        TorchLR.calculate_accuracy(y_train,
                                   r_pred_bn_train.flatten()))
cv_results['r_test_acc'].append(
        TorchLR.calculate_accuracy(y_test,
                                   r_pred_bn_test.flatten()))


if hasattr(torch_model, 'results_df'):
    torch_model.results_df.to_csv(os.path.join(args.results_dir,
                                               'torch_params_{}.tsv'.format(
                                                   args.seed)), sep='\t')

cv_results_file = os.path.join(args.results_dir,
                               'cv_results_{}.pkl'.format(args.seed))
torch_coef_file = os.path.join(args.results_dir,
                               'torch_coefs_{}.tsv.gz'.format(args.seed))

with open(cv_results_file, 'wb') as f:
    pkl.dump(cv_results, f)

np.savetxt(torch_coef_file, torch_weights, fmt='%.5f', delimiter='\t')

