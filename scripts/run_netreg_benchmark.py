import os
import argparse
import subprocess

import sys; sys.path.append('.')
import config as cfg

networks_dir = os.path.join(cfg.repo_root, 'simdata', 'sim_networks')

def run_benchmark_script(n, p, noise_stdev, uncorr_frac, add_frac,
                         remove_frac, seed, results_dir, gpu=False,
                         add_only_uncorr=False, add_single_edge=False,
                         ignore_network=False):

    num_networks = (p // 5) + 1
    script_args = [
        'python',
        os.path.join(cfg.repo_root, 'netreg_benchmark.py'),
        '--results_dir', results_dir,
        '--param_search',
        '--networks_dir', networks_dir,
        '--num_samples', str(n),
        '--num_features', str(p),
        '--noise_stdev', str(noise_stdev),
        '--uncorr_frac', str(uncorr_frac),
        '--add_frac', str(add_frac),
        '--remove_frac', str(remove_frac),
        '--num_networks', str(num_networks),
        '--seed', str(seed),
        '--verbose'
    ]
    if add_only_uncorr:
        script_args.append('--add_only_uncorr')
    if add_single_edge:
        script_args.append('--add_single_edge')
    if gpu:
        script_args.append('--gpu')
    if ignore_network:
        script_args.append('--ignore_network')

    print('Running: {}'.format(' '.join(script_args)))
    subprocess.check_output(script_args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--add_only_uncorr', action='store_true')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--ignore_network', action='store_true')
    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # run script with given parameters
    data_dims = [
       #(n, p)
       (100, 10),
       (100, 100),
       (100, 1000),
       (500, 10),
       (500, 100),
       (500, 1000),
       (1000, 10),
       (1000, 100),
       (1000, 1000)
    ]
    noise_stdev = 0.1
    uncorr_fracs = [0.1, 0.25, 0.5]
    add_fracs = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    remove_fracs = [0.0, 0.1, 0.2, 0.5, 0.75, 0.9]

    results_dir = os.path.join(cfg.repo_root,
                               'simdata',
                               'results',
                               'network_noise_single')

    results_dir_cntl = os.path.join(cfg.repo_root,
                                    'simdata',
                                    'results',
                                    'network_noise_single_control')

    results_dir_nn = os.path.join(cfg.repo_root,
                                  'simdata',
                                  'results',
                                  'network_noise_single_nn')

    noise_stdev = 0.1
    add_frac = 0.0
    remove_frac = 0.0

    for (n, p) in data_dims:
        for uncorr_frac in uncorr_fracs:
            # for seed in range(5):
            for seed in range(10):
                run_benchmark_script(n, p, noise_stdev, uncorr_frac,
                                     add_frac, remove_frac, seed,
                                     results_dir, gpu=args.gpu,
                                     add_only_uncorr=True,
                                     add_single_edge=True)
                run_benchmark_script(n, p, noise_stdev, uncorr_frac,
                                     add_frac, remove_frac, seed,
                                     results_dir_cntl, gpu=args.gpu,
                                     add_only_uncorr=True)
                run_benchmark_script(n, p, noise_stdev, uncorr_frac,
                                     add_frac, remove_frac, seed,
                                     results_dir_nn, gpu=args.gpu,
                                     ignore_network=True)

