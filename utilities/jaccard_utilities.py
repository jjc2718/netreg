import os
import itertools as it
import pandas as pd

def compute_jaccard(v1, v2):
    v1, v2 = set(v1), set(v2)
    intersection = v1.intersection(v2)
    union = v1.union(v2)
    return ((len(intersection) / len(union) if len(union) != 0 else 0),
            len(intersection),
            len(union))

def get_inter_method_similarity(sk_coefs_folds, torch_coefs_folds,
                                seeds, folds, signal='signal'):
    inter_method_sims = []
    for seed in seeds:
        for fold in folds:
            sk_coefs = sk_coefs_folds[signal][seed][fold][0]
            sk_genes = sk_coefs_folds[signal][seed][fold][1]
            sk_nz_coefs = (sk_coefs != 0)
            sk_nz_genes = sk_genes[sk_nz_coefs]
            torch_coefs = torch_coefs_folds[signal][seed][fold][0]
            torch_genes = torch_coefs_folds[signal][seed][fold][1]
            torch_nz_coefs = (torch_coefs != 0)
            torch_nz_genes = torch_genes[torch_nz_coefs]
            inter_method_sims.append(compute_jaccard(set(sk_nz_genes), set(torch_nz_genes))[0])
    return inter_method_sims

def get_intra_method_similarity(sk_coefs_folds, torch_coefs_folds,
                                seeds, folds, signal='signal'):
    intra_method_sims_sk = []
    intra_method_sims_torch = []

    for seed in seeds:
        for f1, f2 in it.combinations(folds, 2):

            # first for scikit-learn
            sk_coefs_f1 = sk_coefs_folds[signal][seed][f1][0]
            sk_genes_f1 = sk_coefs_folds[signal][seed][f1][1]
            sk_coefs_f2 = sk_coefs_folds[signal][seed][f2][0]
            sk_genes_f2 = sk_coefs_folds[signal][seed][f2][1]
            sk_nz_coefs_f1 = (sk_coefs_f1 != 0)
            sk_nz_genes_f1 = sk_genes_f1[sk_nz_coefs_f1]
            sk_nz_coefs_f2 = (sk_coefs_f2 != 0)
            sk_nz_genes_f2 = sk_genes_f2[sk_nz_coefs_f2]
            intra_method_sims_sk.append(compute_jaccard(set(sk_nz_genes_f1), set(sk_nz_genes_f2))[0])

            # then for torch
            torch_coefs_f1 = torch_coefs_folds[signal][seed][f1][0]
            torch_genes_f1 = torch_coefs_folds[signal][seed][f1][1]
            torch_coefs_f2 = torch_coefs_folds[signal][seed][f2][0]
            torch_genes_f2 = torch_coefs_folds[signal][seed][f2][1]
            torch_nz_coefs_f1 = (torch_coefs_f1 != 0)
            torch_nz_genes_f1 = torch_genes_f1[torch_nz_coefs_f1]
            torch_nz_coefs_f2 = (torch_coefs_f2 != 0)
            torch_nz_genes_f2 = torch_genes_f2[torch_nz_coefs_f2]
            intra_method_sims_torch.append(compute_jaccard(set(torch_nz_genes_f1), set(torch_nz_genes_f2))[0])

    return (intra_method_sims_sk, intra_method_sims_torch)
