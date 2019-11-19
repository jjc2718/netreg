"""
Script to validate network regularization implementation on simulated
data, based on TCGA mutation prediction.

Based on 3.classify-with-raw-expression.py (data sources are the same
but output formats are slightly different).

"""
import os
import argparse
import logging
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import config as cfg
import utilities.data_utilities as du
from utilities.classify_pytorch import TorchLR
from tcga_util import (
    get_threshold_metrics,
    extract_coefficients,
    align_matrices,
    check_status
)

p = argparse.ArgumentParser()
p.add_argument('--gene_list', nargs='*', default=None,
               help='<Optional> Provide a list of genes to run\
                     mutation classification for; default is all genes')
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
algorithm = "raw"

# load processed data from TCGA
genes_df, pancan_data = du.load_raw_data(args.gene_list, verbose=args.verbose)

# load our synthetic data
logging.debug('Loading gene expression data...')
rnaseq_train_df = pd.read_csv(cfg.data_dir.joinpath(
                                  'tcga_train_sim_subset.tsv').resolve(),
                              index_col=0, sep='\t')
# scale synthetic data
train_fitted_scaler = MinMaxScaler().fit(rnaseq_train_df)
rnaseq_train_df = pd.DataFrame(
    train_fitted_scaler.transform(rnaseq_train_df),
    columns=rnaseq_train_df.columns,
    index=rnaseq_train_df.index,
)

# track total metrics for each gene in one file
metric_cols = [
    "auroc",
    "aupr",
    "gene_or_cancertype",
    "signal",
    "z_dim",
    "seed",
    "algorithm",
    "data_type",
]

num_genes = len(genes_df)

for gene_idx, gene_series in genes_df.iterrows():

    gene_name = gene_series.gene
    classification = gene_series.classification

    # Create list to store gene specific results
    gene_auc_list = []
    gene_aupr_list = []
    gene_coef_list = []
    gene_metrics_list = []

    # Create directory for the gene
    gene_dir = os.path.join(args.results_dir, "mutation", gene_name)
    os.makedirs(gene_dir, exist_ok=True)

    # Process the y matrix for the given gene or pathway
    y_df = du.load_labels(gene_name, classification, gene_dir, pancan_data)

    signal = 'signal'
    x_train_raw_df = rnaseq_train_df

    gene_features = x_train_raw_df.columns.values
    train_samples, x_train_df, y_train_df = align_matrices(
        x_file_or_df=x_train_raw_df, y=y_df
    )

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
        'torch_train_auroc': [],
        'torch_train_aupr': [],
        'torch_train_acc': [],
        'torch_tune_auroc': [],
        'torch_tune_aupr': [],
        'torch_tune_acc': []
    }

    # get gene expression features (these will be the ones in the network)
    # TODO: probably should calculate these as intersection of node list and
    #       x_df columns once network doesn't cover all genes
    gene_features = np.isin(x_train_df.columns.values, gene_features)

    # Find the best PyTorch model, by cross-validation on train set
    torch_model = TorchLR(torch_params,
                          seed=args.seed,
                          num_iters=cfg.torch_num_iters,
                          num_inner_folds=cfg.torch_num_inner_folds,
                          network_file=args.network_file,
                          network_features=gene_features,
                          use_gpu=args.gpu,
                          verbose=args.verbose)

    losses, preds, preds_bn = torch_model.train_torch_model(x_train_df.values,
                                                            x_train_df.values,
                                                            y_train_df.status.values,
                                                            y_train_df.status.values)
    best_torch_params = torch_model.best_params

    kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=args.seed)

    # just evaluate on same train set (but different splits) for now
    # TODO: maybe this should be stratified eventually
    for fold, (subtrain_ixs, tune_ixs) in enumerate(kf.split(x_train_df.values), 1):

        logging.debug('Evaluating model on fold {} of {}'.format(
            fold, cfg.folds))

        X_subtrain = x_train_df.values[subtrain_ixs]
        X_tune = x_train_df.values[tune_ixs]
        y_subtrain = y_train_df.status.values[subtrain_ixs]
        y_tune = y_train_df.status.values[tune_ixs]

        # Make predictions using torch model
        # To save:
        # - metrics (accuracy, AUROC, AUPR) for each fold
        # - model coefficients (linear layer weights) for each fold
        # - best parameters from search procedure
        _, torch_preds, torch_preds_bn = torch_model.torch_model(
            X_subtrain, X_tune, y_subtrain, y_tune, best_torch_params,
            save_weights=True)

        torch_weights = torch_model.last_weights.flatten()
        torch_pred_train, torch_pred_tune = torch_preds
        torch_pred_bn_train, torch_pred_bn_tune = torch_preds_bn

        torch_train_results = get_threshold_metrics(
            y_subtrain, torch_pred_train, drop=False
        )
        torch_tune_results = get_threshold_metrics(
            y_tune, torch_pred_tune, drop=False
        )

        cv_results['torch_train_auroc'].append(torch_train_results['auroc'])
        cv_results['torch_train_aupr'].append(torch_train_results['aupr'])
        cv_results['torch_tune_auroc'].append(torch_tune_results['auroc'])
        cv_results['torch_tune_aupr'].append(torch_tune_results['aupr'])
        cv_results['torch_train_acc'].append(
                TorchLR.calculate_accuracy(y_subtrain,
                                           torch_pred_bn_train.flatten()))
        cv_results['torch_tune_acc'].append(
                TorchLR.calculate_accuracy(y_tune,
                                           torch_pred_bn_tune.flatten()))

    cv_results_file = os.path.join(args.results_dir,
                                   'cv_results_{}_l{}.pkl'.format(
                                       args.seed, args.l1_penalty))
    # torch_coef_file = os.path.join(args.results_dir,
    #                                'torch_coefs_{}_l{}.tsv.gz'.format(
    #                                    args.seed, args.l1_penalty))

    with open(cv_results_file, 'wb') as f:
        pkl.dump(cv_results, f)

    # torch_coef_df.to_csv(torch_coef_file,
    #                      sep='\t', index=False, compression='gzip',
    #                      float_format="%.5g")


