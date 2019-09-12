"""
Script to compare PyTorch implementation of logistic regression against
previous scikit-learn implementation, to validate that the former is
correct and gives more or less the same results.

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
from utilities.classify_pytorch import TorchLR
from tcga_util import (
    load_pancancer_data,
    load_top_50,
    subset_genes_by_mad,
    get_threshold_metrics,
    extract_coefficients,
    align_matrices,
    process_y_matrix,
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
args = p.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

np.random.seed(args.seed)
algorithm = "raw"

# load data
logging.debug('Loading gene label data...')
genes_df = load_top_50()
if args.gene_list is not None:
    genes_df = genes_df[genes_df['gene'].isin(args.gene_list)]
    genes_df.reset_index(drop=True, inplace=True)

# loading this data from the pancancer repo is very slow, so we
# cache it in a pickle to speed up loading
pancan_fname = os.path.join(cfg.data_dir, 'pancancer_data.pkl')

if os.path.exists(pancan_fname):
    logging.debug('Loading pan-cancer data from cached pickle file...')
    with open(pancan_fname, 'rb') as f:
        pancan_data = pkl.load(f)
else:
    logging.debug('Loading pan-cancer data from repo (warning: slow)...')
    pancan_data = load_pancancer_data()
    with open(pancan_fname, 'wb') as f:
        pkl.dump(pancan_data, f)

(sample_freeze_df,
 mutation_df,
 copy_loss_df,
 copy_gain_df,
 mut_burden_df) = pancan_data

# Load and process X matrix
logging.debug('Loading gene expression data...')
rnaseq_train = (
    os.path.join(cfg.data_dir,
                 'train_tcga_expression_matrix_processed.tsv.gz')
    )
rnaseq_test = (
    os.path.join(cfg.data_dir,
                 'test_tcga_expression_matrix_processed.tsv.gz')
    )

rnaseq_train_df = pd.read_csv(rnaseq_train, index_col=0, sep='\t')
rnaseq_test_df = pd.read_csv(rnaseq_test, index_col=0, sep='\t')

mad_file = os.path.join(cfg.data_dir, 'tcga_mad_genes.tsv')
rnaseq_train_df, rnaseq_test_df = subset_genes_by_mad(
    rnaseq_train_df, rnaseq_test_df, mad_file, cfg.num_features_raw)

# Scale RNAseq matrix the same way RNAseq was scaled for
# compression algorithms
train_fitted_scaler = MinMaxScaler().fit(rnaseq_train_df)
rnaseq_train_df = pd.DataFrame(
    train_fitted_scaler.transform(rnaseq_train_df),
    columns=rnaseq_train_df.columns,
    index=rnaseq_train_df.index,
)

test_fitted_scaler = MinMaxScaler().fit(rnaseq_test_df)
rnaseq_test_df = pd.DataFrame(
    test_fitted_scaler.transform(rnaseq_test_df),
    columns=rnaseq_test_df.columns,
    index=rnaseq_test_df.index,
)

# Track total metrics for each gene in one file
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
    y_mutation_df = mutation_df.loc[:, gene_name]

    # Include copy number gains for oncogenes
    # and copy number loss for tumor suppressor genes (TSG)
    include_copy = True
    if classification == "Oncogene":
        y_copy_number_df = copy_gain_df.loc[:, gene_name]
    elif classification == "TSG":
        y_copy_number_df = copy_loss_df.loc[:, gene_name]
    else:
        y_copy_number_df = pd.DataFrame()
        include_copy = False

    y_df = process_y_matrix(
        y_mutation=y_mutation_df,
        y_copy=y_copy_number_df,
        include_copy=include_copy,
        gene=gene_name,
        sample_freeze=sample_freeze_df,
        mutation_burden=mut_burden_df,
        filter_count=cfg.filter_count,
        filter_prop=cfg.filter_prop,
        output_directory=gene_dir,
        hyper_filter=5,
    )

    model_no = 1

    for signal in ["signal", "shuffled"]:
        if signal == "shuffled":
            # Shuffle training data
            x_train_raw_df = rnaseq_train_df.apply(
                lambda x: np.random.permutation(x.tolist()),
                axis=1,
                result_type="expand",
            )

            x_train_raw_df.columns = rnaseq_train_df.columns
            x_train_raw_df.index = rnaseq_train_df.index

            # Shuffle testing data
            x_test_raw_df = rnaseq_test_df.apply(
                lambda x: np.random.permutation(x.tolist()),
                axis=1,
                result_type="expand",
            )

            x_test_raw_df.columns = rnaseq_test_df.columns
            x_test_raw_df.index = rnaseq_test_df.index

        else:
            x_train_raw_df = rnaseq_train_df
            x_test_raw_df = rnaseq_test_df

        # Now, perform all the analyses for each X matrix
        train_samples, x_train_df, y_train_df = align_matrices(
            x_file_or_df=x_train_raw_df, y=y_df
        )

        test_samples, x_test_df, y_test_df = align_matrices(
            x_file_or_df=x_test_raw_df, y=y_df
        )

        # Train the model
        logging.debug(
            "Training model {} of 2 for gene {} of {} (gene: {}, for raw {} features)".format(
                model_no, gene_idx+1, num_genes, gene_name, signal)
        )

        model_no += 1

        # Find the best PyTorch model, by cross-validation on train set
        torch_model = TorchLR(cfg.torch_param_choices,
                              seed=args.seed,
                              num_iters=cfg.torch_num_iters,
                              num_inner_folds=cfg.torch_num_inner_folds,
                              use_gpu=args.gpu,
                              verbose=args.verbose)

        losses, preds, preds_bn = torch_model.train_torch_model(x_train_df.values,
                                                                x_test_df.values,
                                                                y_train_df.status.values,
                                                                y_test_df.status.values)
        best_torch_params = torch_model.best_params

        if hasattr(torch_model, 'results_df'):
            torch_model.results_df.to_csv('./pytorch_results/torch_params_{}_{}.tsv'.format(
                signal, args.seed), sep='\t')

        # Find the best scikit-learn model
        cv_pipeline, y_pred_train_df, y_pred_test_df, y_cv_df = train_model(
            x_train=x_train_df,
            x_test=x_test_df,
            y_train=y_train_df,
            alphas=cfg.alphas,
            l1_ratios=cfg.l1_ratios,
            n_folds=cfg.folds,
            max_iter=cfg.max_iter,
        )

        # Compare the models on the same CV splits
        # (models might be fit on different CV splits, this is fine but we
        #  want to evaluate them on the same ones)
        cv_results = {
            'sklearn_train_auroc': [],
            'sklearn_train_aupr': [],
            'sklearn_train_acc': [],
            'sklearn_tune_auroc': [],
            'sklearn_tune_aupr': [],
            'sklearn_tune_acc': [],
            'torch_train_auroc': [],
            'torch_train_aupr': [],
            'torch_train_acc': [],
            'torch_tune_auroc': [],
            'torch_tune_aupr': [],
            'torch_tune_acc': []
        }

        kf = KFold(n_splits=cfg.folds, shuffle=True, random_state=args.seed)

        sklearn_coef_df = None
        torch_coef_df = None

        for fold, (subtrain_ixs, tune_ixs) in enumerate(kf.split(x_train_df.values), 1):

            logging.debug('Evaluating models on fold {} of {}'.format(
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

            # Make predictions using sklearn model
            cv_pipeline.fit(X=X_subtrain, y=y_subtrain)
            sklearn_pred_train = cv_pipeline.decision_function(X_subtrain)
            sklearn_pred_tune = cv_pipeline.decision_function(X_tune)
            sklearn_pred_bn_train = cv_pipeline.predict(X_subtrain)
            sklearn_pred_bn_tune = cv_pipeline.predict(X_tune)

            s_coef_df = extract_coefficients(
                cv_pipeline=cv_pipeline,
                feature_names=x_train_df.columns,
                signal=signal,
                z_dim=len(x_train_df.columns),
                seed=args.seed,
                algorithm=algorithm
            )
            s_coef_df['fold'] = fold

            t_coef_df = s_coef_df.copy()
            t_coef_df['weight'] = torch_weights

            if sklearn_coef_df is None:
                sklearn_coef_df = s_coef_df
            else:
                sklearn_coef_df = pd.concat((sklearn_coef_df, s_coef_df))

            if torch_coef_df is None:
                torch_coef_df = t_coef_df
            else:
                torch_coef_df = pd.concat((torch_coef_df, t_coef_df))

            torch_train_results = get_threshold_metrics(
                y_subtrain, torch_pred_train, drop=False
            )
            torch_tune_results = get_threshold_metrics(
                y_tune, torch_pred_tune, drop=False
            )

            sklearn_train_results = get_threshold_metrics(
                y_subtrain, sklearn_pred_train, drop=False
            )
            sklearn_tune_results = get_threshold_metrics(
                y_tune, sklearn_pred_tune, drop=False
            )

            cv_results['torch_train_auroc'].append(torch_train_results['auroc'])
            cv_results['torch_train_aupr'].append(torch_train_results['aupr'])
            cv_results['torch_tune_auroc'].append(torch_tune_results['auroc'])
            cv_results['torch_tune_aupr'].append(torch_tune_results['aupr'])

            cv_results['sklearn_train_auroc'].append(sklearn_train_results['auroc'])
            cv_results['sklearn_train_aupr'].append(sklearn_train_results['aupr'])
            cv_results['sklearn_tune_auroc'].append(sklearn_tune_results['auroc'])
            cv_results['sklearn_tune_aupr'].append(sklearn_tune_results['aupr'])

            def calculate_accuracy(y, y_pred):
                return np.linalg.norm((1 for i in range(len(y)) if y[i] == y_pred[i]), ord=0) / len(y)

            cv_results['torch_train_acc'].append(
                    calculate_accuracy(y_subtrain, torch_pred_bn_train))
            cv_results['torch_tune_acc'].append(
                    calculate_accuracy(y_tune, torch_pred_bn_tune))
            cv_results['sklearn_train_acc'].append(
                    calculate_accuracy(y_subtrain, sklearn_pred_bn_train))
            cv_results['sklearn_tune_acc'].append(
                    calculate_accuracy(y_tune, sklearn_pred_bn_tune))

        with open(os.path.join(args.results_dir, 'cv_results_{}_{}.pkl'.format(signal, args.seed)),
                  'wb') as f:
            pkl.dump(cv_results, f)

        torch_coef_df.to_csv(os.path.join(args.results_dir,
                                          './pytorch_results/torch_coefs_{}_{}.tsv.gz'.format(signal, args.seed)),
                             sep='\t', index=False, compression='gzip',
                             float_format="%.5g")

        sklearn_coef_df.to_csv(os.path.join(args.results_dir,
                                            './pytorch_results/sklearn_coefs_{}_{}.tsv.gz'.format(signal, args.seed)),
                             sep='\t', index=False, compression='gzip',
                             float_format="%.5g")

