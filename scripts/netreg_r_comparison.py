import os
import subprocess

import sys; sys.path.append('.')
import config as cfg

# just hardcoding PyTorch/R parameters for now
batch_size = 30
learning_rate = 0.003
num_epochs = 500
l1_penalty = 0
network_penalty = 0.1

networks_dir = os.path.join(cfg.repo_root, 'simdata', 'sim_networks')
results_dir = os.path.join(cfg.repo_root, 'simdata', 'clique_results')

def run_benchmark_script(n, p, noise_stdev, uncorr_frac, seed):

    num_networks = (p // 5) + 1
    script_args = [
        'python',
        os.path.join(cfg.repo_root, 'netreg_benchmark.py'),
        '--results_dir', results_dir,
        '--batch_size', str(batch_size),
        '--learning_rate', str(learning_rate),
        '--num_epochs', str(num_epochs),
        '--l1_penalty', str(l1_penalty),
        '--network_penalty', str(network_penalty),
        '--networks_dir', networks_dir,
        '--num_samples', str(n),
        '--num_features', str(p),
        '--noise_stdev', str(noise_stdev),
        '--uncorr_frac', str(uncorr_frac),
        '--num_networks', str(num_networks),
        '--seed', str(seed),
        '--verbose'
    ]
    print('Running: {}'.format(' '.join(script_args)))
    subprocess.check_output(script_args)

# run script with given parameters
data_dims = [
   #(n, p)
   (100, 10),
   (100, 100),
   (100, 1000),
   # (500, 10),
   # (500, 100),
   # (500, 1000),
   # (1000, 10),
   # (1000, 100),
   # (1000, 1000)
]
noise_stdevs = [0, 0.1, 1, 10]
fracs = [0, 0.25, 0.5, 0.75, 1.0]

for (n, p) in data_dims:
    for noise_stdev in noise_stdevs:
        for uncorr_frac in fracs:
            for seed in range(5):
                run_benchmark_script(n, p, noise_stdev, uncorr_frac, seed)
