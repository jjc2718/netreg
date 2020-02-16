import pathlib

# home is where the file is
repo_root = pathlib.Path(__file__).parents[0]

# important subdirectories
data_dir = repo_root.joinpath('data').resolve()
networks_dir = data_dir.joinpath('networks').resolve()
pathway_data = data_dir.joinpath('pathway_data').resolve()
models_dir = repo_root.joinpath('compression_models').resolve()
results_dir = repo_root.joinpath('results').resolve()
scripts_dir = repo_root.joinpath('scripts').resolve()

# location of saved expression data
pancan_data = data_dir.joinpath('pancancer_data.pkl').resolve()
mad_data = data_dir.joinpath('tcga_mad_genes.tsv').resolve()
rnaseq_train = data_dir.joinpath(
                    'train_tcga_expression_matrix_processed.tsv.gz').resolve()
rnaseq_test = data_dir.joinpath(
                    'test_tcga_expression_matrix_processed.tsv.gz').resolve()

# parameters for classification using raw gene expression
num_features_raw = 8000

# hyperparameters for classification experiments
filter_prop = 0.05
filter_count = 15
folds = 3
max_iter = 200
alphas = [0.1, 0.13, 0.15, 0.2, 0.25, 0.3]
l1_ratios = [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]

# location of saved classify results, for regression testing
fixtures_dir = repo_root.joinpath('tests').joinpath('fixtures').resolve()
saved_results_train = fixtures_dir.joinpath('saved_results_train.tsv.gz').resolve()
saved_results_test = fixtures_dir.joinpath('saved_results_test.tsv.gz').resolve()
saved_coefs = fixtures_dir.joinpath('saved_coefs.tsv.gz').resolve()

default_seed = 42

# data generation parameters for regression tests
test_params = {
    'n_train': 100,
    'n_test': 100,
    'p': 200
}
test_size = 0.2

# hyperparameters for PyTorch logistic regression
torch_param_choices = {
    'learning_rate': [0.001, 0.0001, 5e-5, 1e-5],
    'batch_size': [20, 50, 100],
    'num_epochs': [100, 200, 500],
    'l1_penalty': [0.01, 0.05, 0.1]
}
torch_num_iters = 10
torch_num_inner_folds = 3

# hyperparameters for network-regularized logistic regression
netreg_param_choices = {
    'learning_rate': [0.01, 0.005, 0.001, 5e-4],
    'batch_size': [50],
    'num_epochs': [100, 200, 500],
    'l1_penalty': [0.0],
    'network_penalty': [0, 0.1, 1, 10, 100]
}
