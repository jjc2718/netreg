import pathlib

repo_root = pathlib.Path()

data_dir = repo_root.joinpath('data').resolve()
pathway_data = data_dir.joinpath('pathway_data').resolve()
models_dir = repo_root.joinpath('models').resolve()
results_dir = repo_root.joinpath('results').resolve()
scripts_dir = repo_root.joinpath('scripts').resolve()

default_seed = 42

# hyperparameters for classification experiments
filter_prop = 0.05
filter_count = 15
folds = 3
max_iter = 100
alphas = [0.1, 0.13, 0.15, 0.2, 0.25, 0.3]
l1_ratios = [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]

# parameters for classification using raw gene expression
num_features_raw = 8000

# hyperparameters for PyTorch logistic regression
torch_param_choices = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'batch_size': [10, 20, 50, 100],
    'num_epochs': [100, 200, 500],
    'l1_penalty': [0, 0.01, 0.1, 1, 10]
}
torch_num_iters = 10
torch_num_inner_folds = 3
