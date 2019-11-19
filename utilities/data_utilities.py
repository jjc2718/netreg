import os
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler

import config as cfg
from tcga_util import process_y_matrix

def load_raw_data(gene_list, verbose=False):
    # load data
    if verbose:
        print('Loading gene label data...')

    genes_df = load_top_50()
    if gene_list is not None:
        genes_df = genes_df[genes_df['gene'].isin(gene_list)]
        genes_df.reset_index(drop=True, inplace=True)

    # loading this data from the pancancer repo is very slow, so we
    # cache it in a pickle to speed up loading
    if os.path.exists(cfg.pancan_data):
        if verbose:
            print('Loading pan-cancer data from cached pickle file...')
        with open(cfg.pancan_data, 'rb') as f:
            pancan_data = pkl.load(f)
    else:
        if verbose:
            print('Loading pan-cancer data from repo (warning: slow)...')
        pancan_data = load_pancancer_data_from_repo()
        with open(cfg.pancan_data, 'wb') as f:
            pkl.dump(pancan_data, f)

    return (genes_df, pancan_data)


def load_expression_data(subset_mad_genes=cfg.num_features_raw,
                         scale_input=False, verbose=False):
    # Load and process X matrix
    if verbose:
        print('Loading gene expression data...')

    rnaseq_train_df = pd.read_csv(cfg.rnaseq_train, index_col=0, sep='\t')
    rnaseq_test_df = pd.read_csv(cfg.rnaseq_test, index_col=0, sep='\t')

    if subset_mad_genes is not None:
        rnaseq_train_df, rnaseq_test_df = subset_genes_by_mad(
            rnaseq_train_df, rnaseq_test_df, cfg.mad_data, subset_mad_genes)

    # Scale RNAseq matrix the same way RNAseq was scaled for
    # compression algorithms
    if scale_input:
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

    return (rnaseq_train_df, rnaseq_test_df)


def load_top_50():
    """Load top 50 mutated genes in TCGA from BioBombe repo.

    These were precomputed for the equivalent experiments in the
    BioBombe paper, so no need to recompute them.
    """
    base_url = "https://github.com/greenelab/BioBombe/raw"
    commit = "aedc9dfd0503edfc5f25611f5eb112675b99edc9"

    file = "{}/{}/9.tcga-classify/data/top50_mutated_genes.tsv".format(
            base_url, commit)
    genes_df = pd.read_csv(file, sep='\t')
    return genes_df


def load_pancancer_data_from_repo():
    """Load data to build feature matrices from pancancer repo."""

    base_url = "https://github.com/greenelab/pancancer/raw"
    commit = "2a0683b68017fb226f4053e63415e4356191734f"

    file = "{}/{}/data/sample_freeze.tsv".format(base_url, commit)
    sample_freeze_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/pancan_mutation_freeze.tsv.gz".format(base_url, commit)
    mutation_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_loss_status.tsv.gz".format(base_url, commit)
    copy_loss_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/copy_number_gain_status.tsv.gz".format(base_url, commit)
    copy_gain_df = pd.read_csv(file, index_col=0, sep='\t')

    file = "{}/{}/data/mutation_burden_freeze.tsv".format(base_url, commit)
    mut_burden_df = pd.read_csv(file, index_col=0, sep='\t')

    return (
        sample_freeze_df,
        mutation_df,
        copy_loss_df,
        copy_gain_df,
        mut_burden_df
    )


def subset_genes_by_mad(train_df, test_df, mad_file, subset_mad_genes):
    """Subset genes by mean absolute deviation."""

    mad_genes_df = pd.read_csv(mad_file, sep='\t')
    mad_genes = mad_genes_df.iloc[0:subset_mad_genes, ].gene_id.astype(str)

    train_df = train_df.reindex(mad_genes, axis='columns')
    test_df = test_df.reindex(mad_genes, axis='columns')
    return (train_df, test_df)


def load_labels(gene_name, classification, gene_dir, pancan_data,
                include_copy=True):
    """Load classification labels using processed TCGA data."""

    # unpack pancancer data
    (sample_freeze_df,
     mutation_df,
     copy_loss_df,
     copy_gain_df,
     mut_burden_df) = pancan_data

    # process the y matrix for the given gene or pathway
    y_mutation_df = mutation_df.loc[:, gene_name]

    # include copy number gains for oncogenes
    # and copy number loss for tumor suppressor genes (TSG)
    if classification == "Oncogene":
        y_copy_number_df = copy_gain_df.loc[:, gene_name]
    elif classification == "TSG":
        y_copy_number_df = copy_loss_df.loc[:, gene_name]
    else:
        y_copy_number_df = pd.DataFrame()
        include_copy = False

    # assemble and return labels
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

    return y_df

