"""
Utilities for processing and comparing latent spaces

"""
import os
import glob
import numpy as np
import pandas as pd

def get_overlap_cols_from_plier(models_dirs):
    # this implements approach 1 from 5.analyze_plier_compression.ipynb
    genesets = []
    for m_dir in models_dirs:
        plier_pattern = os.path.join(m_dir,
                                     'components_10',
                                     'plier_*_weight_matrix.tsv.gz')
        fnames = glob.glob(plier_pattern)
        genesets.append(set(pd.read_csv(fnames[0], sep='\t', index_col=0).columns.values))
    return sorted(list(genesets[0].intersection(*genesets[1:])))

def get_overlap_cols_from_files(f1, f2):
    # this implements approach 2 from 5.analyze_plier_compression.ipynb
    f1_names = set(pd.read_csv(f1, sep='\t', index_col=0).columns.values)
    f2_names = set(pd.read_csv(f2, sep='\t', index_col=0).columns.values)
    return sorted(list(f1_names.intersection(f2_names)))

def get_matrices_from_files(files, gene_subset, shuffled=False):
    mtxs, filenames = [], []
    if shuffled:
        files = [f for f in files if 'shuffled' in f]
    else:
        files = [f for f in files if 'shuffled' not in f]
    for f in files:
        mtxs.append(pd.read_csv(f, sep='\t', header=0, index_col=0)[gene_subset])
        filenames.append(f)
    return (mtxs, files)

def calculate_avg_cca(z_dims, models_map, overlap=False, verbose=False):
    import itertools
    import utilities.cca_core as cca_core

    algorithms = list(models_map.keys())
    avg_cca_mtx = {z_dim: np.zeros((len(algorithms), len(algorithms))) for z_dim in z_dims}

    for z_dim in z_dims:
        for alg1, alg2 in itertools.combinations_with_replacement(algorithms, 2):
            if verbose:
                print('Comparing {} with {} for z={}...'.format(alg1, alg2, z_dim), end='')
            i1, i2 = algorithms.index(alg1), algorithms.index(alg2)
            cca_values = []
            alg1_pattern = os.path.join(models_map[alg1],
                        'components_{}'.format(z_dim),
                        '{}_*_weight_matrix.tsv.gz'.format(alg1.split('_')[0]))
            alg2_pattern = os.path.join(models_map[alg2],
                        'components_{}'.format(z_dim),
                        '{}_*_weight_matrix.tsv.gz'.format(alg2.split('_')[0]))
            alg1_files = glob.glob(alg1_pattern)
            alg2_files = glob.glob(alg2_pattern)
            if overlap:
                overlap_cols = get_overlap_cols_from_files(alg1_files[0],
                                                              alg2_files[0])
            else:
                overlap_cols = get_overlap_cols_from_plier(
                                             list(set(models_map.values())))
            (alg1_matrices, alg1_files) = get_matrices_from_files(alg1_files,
                                                                     overlap_cols)
            (alg2_matrices, alg2_files) = get_matrices_from_files(alg2_files,
                                                                     overlap_cols)
            for s1, s2 in itertools.product(range(len(alg1_matrices)),
                                            range(len(alg2_matrices))):
                cca_result = cca_core.robust_cca_similarity(alg1_matrices[s1],
                                                            alg2_matrices[s2],
                                                            verbose=False)
                cca_values.append(np.mean(cca_result['mean']))
            avg_cca_mtx[z_dim][i1, i2] = np.mean(cca_values)
            avg_cca_mtx[z_dim][i2, i1] = avg_cca_mtx[z_dim][i1, i2]
            if verbose:
                print('done')

    return avg_cca_mtx

