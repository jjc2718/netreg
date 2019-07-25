"""
Script to run mutation detection pipeline

"""
import pathlib
import subprocess

import sys; sys.path.append('.')
import config as cfg
from data_models import DataModel

pathway_map = {
    cfg.pathway_data.joinpath('canonical_mapped.tsv').resolve(): 'canonical_pathways',
    cfg.pathway_data.joinpath('oncogenic_mapped.tsv').resolve(): 'oncogenic_pathways',
    cfg.pathway_data.joinpath('randomized_pathways.tsv').resolve(): 'random_pathways'
}

k_vals = [10, 20, 50, 100, 200]
algorithms = DataModel.list_algorithms()
# gene list from BioBombe paper, just do these for now
genes = ['TP53', 'PTEN', 'PIK3CA', 'KRAS', 'TTN']

def run_compression(algorithm, k):
    if alg == 'plier':
        for pathway_file, pathway_dir in pathway_map.items():
            cmd = ['python', '1.compress_given_z.py',
                   '-a', alg, '-k', str(k), '-v',
                   '-p', str(pathway_file),
                   '-o', str(cfg.models_dir.joinpath(pathway_dir).resolve())]
            print('Running: {}'.format(' '.join(cmd)))
            # subprocess.check_call(cmd)
            cmd = ['python', '1.compress_given_z.py',
                   '-a', alg, '-k', str(k), '-s', '-v',
                   '-p', str(pathway_file),
                   '-o', str(cfg.models_dir.joinpath(pathway_dir).resolve())]
            print('Running: {}'.format(' '.join(cmd)))
            # subprocess.check_call(cmd)
    else:
        cmd = ['python', '1.compress_given_z.py',
               '-a', alg, '-k', str(k), '-v',
               '-o', str(cfg.models_dir.joinpath('canonical_pathways').resolve())]
        print('Running: {}'.format(' '.join(cmd)))
        # subprocess.check_call(cmd)
        cmd = ['python', '1.compress_given_z.py',
               '-a', alg, '-k', str(k), '-s', '-v',
               '-o', str(cfg.models_dir.joinpath('canonical_pathways').resolve())]
        print('Running: {}'.format(' '.join(cmd)))
        # subprocess.check_call(cmd)

# first run compression step
for k in k_vals:
    for alg in algorithms:
        run_compression(alg, k)

# then run classification step using compressed models
for pathway_dir in pathway_map.values():
    cmd = ['python', '2.classify_mutations.py',
            '--v', '--g', ' '.join(genes),
            '--m', str(cfg.models_dir.joinpath(pathway_dir).resolve()),
            '--r', str(cfg.results_dir.joinpath(pathway_dir).resolve())]
    print('Running: {}'.format(' '.join(cmd)))
    # subprocess.check_call(cmd)

# then run classification using raw expression values as a baseline
cmd = ['python', '3.classify_with_raw_expression.py',
        '--v', '--g', ' '.join(genes),
        '--r', str(cfg.results_dir.joinpath('canonical_pathways').resolve())]
print('Running: {}'.format(' '.join(cmd)))
# subprocess.check_call(cmd)


