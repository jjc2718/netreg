"""
Script to run cross-validation, holding out cancer types.

"""
import subprocess
import pathlib

import sys; sys.path.append('.')
import config as cfg

results_dir = cfg.results_dir.joinpath('holdout_cancer_type').resolve()

# cancer types to hold out for now
# TODO come up with a systematic way to try all that are possible
# (i.e. have enough samples) for each gene, this is variable between genes
gene_cancer_map = {
    'TP53': ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KICH', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'UCEC', 'UCS'],
    'PTEN': ['BLCA', 'BRCA', 'CESC', 'COAD', 'GBM', 'HNSC', 'LGG', 'LIHC', 'LUSC', 'PRAD', 'SARC', 'SKCM', 'STAD', 'UCEC'],
    'PIK3CA': ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'HNSC', 'LGG', 'LUAD', 'LUSC', 'OV', 'READ', 'SKCM', 'STAD', 'UCEC', 'UCS'],
    'KRAS': ['BLCA', 'CESC', 'COAD', 'LUAD', 'OV', 'PAAD', 'READ', 'STAD', 'TGCT', 'UCEC'],
    'TTN': ['BLCA', 'BRCA', 'CESC', 'COAD', 'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'UCEC'] ,
}

num_genes = len(gene_cancer_map.keys())
for i, (gene, cancer_types) in enumerate(gene_cancer_map.items(), 1):
    num_cancer_types = len(cancer_types)
    for j, holdout_cancer_type in enumerate(cancer_types, 1):
        print('Running: gene {} ({}/{}), cancer type {} ({}/{})'.format(
            gene, i, num_genes, holdout_cancer_type, j, num_cancer_types))
        args = [
            'python',
            'classify_holdout_type.py',
            '--gene', gene,
            '--holdout_cancer_type', holdout_cancer_type,
            '--results_dir', results_dir
        ]
        subprocess.check_call(args)

