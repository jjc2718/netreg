import time
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    cross_val_predict
)
import torch
import torch.nn as nn
import torch.utils.data as data_utils

class LogisticRegression(nn.Module):

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # one output for binary classification
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

def get_params_map(param_choices, num_iters=10, seed=1):
    """Get random combinations of hyperparameters to search over.

    TODO more documentation needed
    """
    import random; random.seed(seed)
    # sorting here ensures that results for models that share the same
    # parameters will have the same choices, and thus will be easily
    # comparable
    param_options = sorted(param_choices.items())
    params_map = {p: [random.choice(vals) for _ in range(num_iters)]
                     for p, vals in param_options}
    return params_map

def torch_param_selection(X_train,
                          y_train,
                          params_map,
                          num_folds,
                          seed=1,
                          use_gpu=False,
                          verbose=False):
    """do the thing.

    Dataset terminology: (subtrain | tune) = train | test
    (avoiding the term "validation" since it's overloaded in biology)
    """
    # k-fold cross-validation over the training data
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    results_df = None
    for fold, (subtrain_ixs, tune_ixs) in enumerate(kf.split(X_train), 1):
        X_subtrain, X_tune = X_train[subtrain_ixs], X_train[tune_ixs]
        y_subtrain, y_tune = y_train[subtrain_ixs], y_train[tune_ixs]
        if verbose:
            print('Running CV fold {} of {}'.format(fold, num_folds))
        result_df = torch_tuning(X_subtrain, X_tune,
                                 y_subtrain, y_tune,
                                 params_map, use_gpu,
                                 verbose)
        result_df['fold'] = fold
        if results_df is None:
            results_df = result_df
        else:
            results_df = pd.concat((results_df, result_df), ignore_index=True)

    sorted_df = (
        results_df.loc[results_df['train/tune'] == 'tune']
                  .groupby('param_set')
                  .mean()
                  .reset_index()
                  .sort_values(by='loss')
    )
    best_ix = sorted_df.loc[0, 'param_set']
    best_params = {k: v[best_ix] for k, v in params_map.items()}
    return results_df, best_params


def torch_tuning(X_subtrain, X_tune, y_subtrain, y_tune, params_map,
                 use_gpu=False, verbose=False):
    """Run parameter search on a single subtrain/tune split."""
    result = {
        'param_set': [],
        'train/tune': [],
        'loss': []
    }
    num_iters = len(params_map[list(params_map.keys())[0]])
    for ix in range(num_iters):
        if verbose:
            print('-- Running parameter set {} of {}...'.format(ix+1, num_iters),
                  end='')
        params = {k: v[ix] for k, v in params_map.items()}
        losses, _, __ = torch_model(X_subtrain, X_tune,
                                    y_subtrain, y_tune,
                                    params, use_gpu,
                                    verbose)
        subtrain_loss, tune_loss = losses
        if verbose:
            print('train_loss: {:.4f}, tune_loss: {:.4f}'.format(subtrain_loss, tune_loss))
        result['param_set'].append(ix)
        result['train/tune'].append('train')
        result['loss'].append(subtrain_loss)
        result['param_set'].append(ix)
        result['train/tune'].append('tune')
        result['loss'].append(tune_loss)
    return pd.DataFrame(result)


def torch_model(X_train, X_test,
                y_train, y_test,
                params, use_gpu=False,
                verbose=False):

    if verbose:
        t = time.time()

    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    l1_penalty = params['l1_penalty']

    if use_gpu:
        X_tr = torch.stack([torch.Tensor(x).cuda() for x in X_train])
        X_ts = torch.stack([torch.Tensor(x).cuda() for x in X_test])
        y_tr = torch.Tensor(y_train).view(-1, 1).cuda()
        y_ts = torch.Tensor(y_test).view(-1, 1).cuda()
    else:
        X_tr = torch.stack([torch.Tensor(x) for x in X_train])
        X_ts = torch.stack([torch.Tensor(x) for x in X_test])
        y_tr = torch.Tensor(y_train).view(-1, 1)
        y_ts = torch.Tensor(y_test).view(-1, 1)

    train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_tr, y_tr),
            batch_size=batch_size, shuffle=True)

    model = LogisticRegression(X_train.shape[1])
    if use_gpu:
        model = model.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            l1_loss = sum(torch.norm(param, 1) for param in model.parameters())
            loss += l1_penalty * l1_loss
            running_loss += loss
            loss.backward()
            optimizer.step()
        scheduler.step(running_loss)

    y_pred_train = model(X_tr)
    y_pred_test = model(X_ts)

    train_loss = float((
            criterion(y_pred_train, y_tr) +
            l1_penalty * sum(torch.norm(param, 1) for param in model.parameters())
    ).detach())

    test_loss = float((
            criterion(y_pred_test, y_ts) +
            l1_penalty * sum(torch.norm(param, 1) for param in model.parameters())
    ).detach())

    if verbose:
        print('Time: {:.4f}'.format(time.time() - t))

    y_pred_train = y_pred_train.detach().numpy()
    y_pred_test = y_pred_test.detach().numpy()

    y_pred_bn_train = (y_pred_train > 0.5).astype('int')
    y_pred_bn_test = (y_pred_test > 0.5).astype('int')

    return ((train_loss, test_loss),
            (y_pred_train, y_pred_test),
            (y_pred_bn_train, y_pred_bn_test))


def train_torch_model(X_train, X_test,
                      y_train, y_test,
                      params_map,
                      num_inner_folds=4,
                      seed=1,
                      use_gpu=False,
                      verbose=False):

    min_params_length = min(len(vs) for k, vs in params_map.items())
    if min_params_length > 1:
        # if multiple hyperparameter choices are provided, get the best
        # set of hyperparameters from a random search
        results_df, best_params = torch_param_selection(X_train, y_train,
                                                        params_map,
                                                        num_inner_folds,
                                                        seed=seed,
                                                        use_gpu=use_gpu,
                                                        verbose=verbose)
    else:
        # else just use the hyperparameters provided (since there's only
        # one choice)
        best_params = {k: vs[0] for k, vs in params_map.items()}

    losses, preds, preds_bn = torch_model(X_train, X_test, y_train, y_test,
                                          best_params, use_gpu=use_gpu,
                                          verbose=verbose)

    return losses, preds, preds_bn


if __name__ == '__main__':
    # TODO: make sure refactor works, etc on small datasets
    # then try it on mutation detection data (train/eval on train set, don't
    # touch test set until sure new code works)

    import argparse

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    import sys; sys.path.append('.')
    import config as cfg
    from tcga_util import get_threshold_metrics

    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # hyperparameter choices to do a random search over
    '''

    sklearn_param_choices = {
        'alpha': [0.1, 0.13, 0.15, 0.2, 0.25, 0.3],
        'l1_ratio': [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]
    }

    # get num_iters different combinations of hyperparameters to test
    sklearn_params_map = get_params_map(sklearn_param_choices,
                                        num_iters=num_iters,
                                        seed=args.seed)
    '''
    torch_param_choices = {
        'learning_rate': [0.005, 0.001, 0.0001, 0.00005],
        'batch_size': [10, 20, 50, 100],
        'num_epochs': [200, 500, 1000],
        'l1_penalty': [0, 0.01, 0.1, 1, 10]
    }
    num_iters = 5
    torch_params_map = get_params_map(torch_param_choices,
                                      num_iters=num_iters,
                                      seed=args.seed)

    '''
    torch_params_map = {
        'learning_rate': [0.0001],
        'batch_size': [10],
        'num_epochs': [200],
        'l1_penalty': [0]
    }
    '''

    # load data and split into train/test sets
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=args.seed)

    # classify using sklearn SGDClassifier
    '''
    y_pred = train_sklearn_model(X_train, X_test,
                                 y_train,
                                 cfg.alphas,
                                 cfg.l1_ratios,
                                 seed=args.seed)

    y_pred_train, y_pred_test, y_pred_bn_train, y_pred_bn_test = y_pred

    sk_train_acc = sum(
            [1 for i in range(len(y_pred_train))
               if y_pred_bn_train[i] == y_train[i]]
    ) / len(y_pred_train)
    sk_test_acc = sum(
            [1 for i in range(len(y_pred_test))
               if y_pred_bn_test[i] == y_test[i]]
    ) / len(y_pred_test)

    sk_train_results = get_threshold_metrics(y_train, y_pred_train)
    sk_test_results = get_threshold_metrics(y_test, y_pred_test)
    '''

    losses, preds, preds_bn = train_torch_model(X_train, X_test,
                                                y_train, y_test,
                                                torch_params_map,
                                                use_gpu=args.gpu,
                                                verbose=args.verbose)

    y_pred_train, y_pred_test = preds
    y_pred_bn_train, y_pred_bn_test = preds_bn

    def calculate_accuracy(y, y_pred):
        return sum(1 for i in range(len(y)) if y[i] == y_pred[i]) / len(y)

    torch_train_acc = calculate_accuracy(y_train, y_pred_bn_train)
    torch_test_acc = calculate_accuracy(y_test, y_pred_bn_test)

    torch_train_results = get_threshold_metrics(y_train, y_pred_train)
    torch_test_results = get_threshold_metrics(y_test, y_pred_test)

    '''
    print('Sklearn train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        sk_train_acc, sk_test_acc))
    print('Sklearn train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        sk_train_results['auroc'], sk_test_results['auroc']))
    print('Sklearn train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        sk_train_results['aupr'], sk_test_results['aupr']))
    '''

    print('Torch train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        torch_train_acc, torch_test_acc))
    print('Torch train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        torch_train_results['auroc'], torch_test_results['auroc']))
    print('Torch train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        torch_train_results['aupr'], torch_test_results['aupr']))

    y_pred_train = np.random.uniform(size=(len(y_train),))
    y_pred_test = np.random.uniform(size=(len(y_test),))
    y_pred_bn_train = (y_pred_train > 0.5).astype('int')
    y_pred_bn_test = (y_pred_test > 0.5).astype('int')

    random_train_acc = calculate_accuracy(y_train, y_pred_bn_train)
    random_test_acc = calculate_accuracy(y_test, y_pred_bn_test)

    random_train_results = get_threshold_metrics(y_train, y_pred_train)
    random_test_results = get_threshold_metrics(y_test, y_pred_test)

    print('Random guessing train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        random_train_acc, random_test_acc))
    print('Random guessing train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        random_train_results['auroc'], random_test_results['auroc']))
    print('Random guessing train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        random_train_results['aupr'], random_test_results['aupr']))
