import os
import argparse
import subprocess

import sys; sys.path.append('.')
import config as cfg

networks_dir = os.path.join(cfg.repo_root, 'simdata', 'sim_networks')

def run_benchmark_script(n, p, noise_stdev, uncorr_frac, seed,
                         results_dir, gpu=False, ignore_network=False):

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
        '--num_networks', str(num_networks),
        '--seed', str(seed),
        '--verbose'
    ]
    if gpu:
        script_args.append('--gpu')
    if ignore_network:
        script_args.append('--ignore_network')

    print('Running: {}'.format(' '.join(script_args)))
    subprocess.check_output(script_args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--ignore_network', action='store_true')
    return p.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # run script with given parameters
    data_dims = [
       #(n, p)
       # (100, 10),
       # (100, 100),
       # (100, 1000),
       (500, 10),
       (500, 100),
       (500, 1000),
       # (1000, 10),
       # (1000, 100),
       # (1000, 1000)
    ]
    noise_stdevs = [0, 0.1, 1, 10]
    fracs = [0, 0.25, 0.5, 0.75, 1.0]

    if args.ignore_network:
        results_dir = os.path.join(cfg.repo_root, 'simdata', 'param_search_results', 'ignore_network')
    else:
        # results_dir = os.path.join(cfg.repo_root, 'simdata', 'param_search_results')
        results_dir = os.path.join(cfg.repo_root, 'test_benchmark')

    for (n, p) in data_dims:
        for noise_stdev in noise_stdevs:
            for uncorr_frac in fracs:
                for seed in range(5):
                    run_benchmark_script(n, p, noise_stdev, uncorr_frac, seed,
                                         results_dir, gpu=args.gpu,
                                         ignore_network=args.ignore_network)
