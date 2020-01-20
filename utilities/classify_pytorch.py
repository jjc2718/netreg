import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    cross_val_predict
)
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.utils.data as data_utils


class LogisticRegression(nn.Module):
    """Model for PyTorch logistic regression."""

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        # one output for binary classification
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

class TorchLR:
    """Class to run hyperparameter search/cross-validation.

    Maintains state for training/cross-validation results.
    """
    def __init__(self,
                 params_map,
                 seed=1,
                 num_iters=10,
                 num_inner_folds=4,
                 network_file=None,
                 network_features=None,
                 learning_curves=False,
                 use_gpu=False,
                 verbose=False):

        # set random seeds
        # https://pytorch.org/docs/stable/notes/randomness.html
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if use_gpu and torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # if both network_file and network_penalty are provided, load network
        # data and calculate graph Laplacian
        if network_file is not None and params_map['network_penalty'] != 0:
            assert network_features is not None, (
                'list of features in network should be included')
            self.network_features = network_features
            import networkx as nx
            G = nx.read_weighted_edgelist(network_file, delimiter='\t')
            # TODO: this should probably be stored as a sparse matrix
            lt = nx.laplacian_matrix(G)
            indices, values, shape = self._convert_csr_to_sparse_inputs(lt)
            self.laplacian = torch.sparse.FloatTensor(indices, values, shape)
        else:
            self.laplacian = None

        # dict for monitoring quantities of interest (e.g. loss over epochs)
        self.monitor_ = {
            'train_loss': [],
            'test_loss': [],
            'l1_loss': [],
            'network_loss': []
        }

        max_params_length = max(len(vs) for k, vs in params_map.items())
        # if there's only one choice provided for each hyperparameter,
        # we'll skip the parameter search later
        #
        # if there are multiple choices, select num_iters parameter
        # combinations to be tested during the parameter search
        if max_params_length > 1:
            params_map = get_params_map(params_map,
                                        self.seed,
                                        num_iters=num_iters)
        self.params_map = params_map
        self.num_inner_folds = num_inner_folds
        self.use_gpu = use_gpu
        self.learning_curves = learning_curves
        self.verbose = verbose


    def _convert_csr_to_sparse_inputs(self, X):
        # adapted from code at https://github.com/suinleelab/attributionpriors
        import scipy.sparse as sp
        coo = sp.coo_matrix(X)
        indices = torch.LongTensor(np.mat([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        return indices, values, coo.shape


    @staticmethod
    def calculate_accuracy(y, y_pred):
        """Calculate accuracy given true labels and predicted labels."""
        assert (y.ndim == 1 and y_pred.ndim == 1), "labels must be flattened"
        return (y == y_pred).mean()


    @staticmethod
    def get_params_map(param_choices, seed, num_iters=10):
        """Get random combinations of hyperparameters to search over.

        Currently combinations are selected with replacement, i.e. duplicates can
        happen.

        TODO: could make this sample from continuous distributions too,
        might be useful for some params

        Parameters
        ----------
        param_choices: dict, (str: list)
            Maps hyperparameter names to choices (currently only works with
            discrete values). Example:
            param_choices = {
                'learning_rate': [0.005, 0.001, 0.0001, 0.00005],
                'batch_size': [10, 20, 50, 100],
                'num_epochs': [200, 500, 1000],
                'l1_penalty': [0, 0.01, 0.1, 1, 10]
            }

        num_iters : int
            The number of combinations to search over.

        Returns
        -------
        dict, (str: list)
            Maps hyperparameter names to lists of values to try.

        """
        import random; random.seed(seed)
        # sorting here ensures that results for models that share the same
        # parameters will have the same choices, and thus will be easily
        # comparable
        param_options = sorted(param_choices.items())
        params_map = {p: [random.choice(vals) for _ in range(num_iters)]
                         for p, vals in param_options}
        return params_map


    def train_torch_model(self, X_train, X_test, y_train, y_test,
                          save_weights=False):
        """Wrapper function for PyTorch model training.

        If multiple hyperparameter choices are provided, get the best
        set of hyperparameters from a random search. Otherwise, just use
        the hyperparameters provided to train/evaluate the model.
        """
        max_params_length = max(len(vs) for k, vs in self.params_map.items())
        if max_params_length > 1:
            results_df, best_params = self.torch_param_selection(X_train, y_train)
            self.results_df = results_df
        else:
            best_params = {k: vs[0] for k, vs in self.params_map.items()}

        self.best_params = best_params

        losses, preds, preds_bn = self.torch_model(X_train, X_test, y_train, y_test,
                                                   best_params,
                                                   save_weights=save_weights,
                                                   learning_curves=self.learning_curves)

        return losses, preds, preds_bn


    def torch_model(self, X_train, X_test, y_train, y_test, params,
                    save_weights=False, learning_curves=False):

        """Main function for training PyTorch model.

        Parameters
        ----------
        X_train : array_like, [n_samples, n_features]
            Training data

        X_test : array_like, [n_samples, n_features]
            Data to evaluate model on

        y_train : array_like, [n_samples]
            Training labels

        y_test : array_like, [n_samples]
            Labels to evaluate model on

        params : dict, (str: mixed)
            Maps hyperparameter names to a single value, used to train the model

        save_weights: bool
            Whether or not to save weights (coefficients) from trained model

        Returns
        -------
        tuple : ((list, list), (list, list), (list, list))
            ((loss on training data, loss on test data),
             (predictions on training data, predictions on testing data),
             (binarized predictions on training/test data))
        """
        # TODO: make this a class option
        classify = False

        if self.verbose:
            t = time.time()

        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        num_epochs = params['num_epochs']
        l1_penalty = params['l1_penalty']

        if self.laplacian is not None:
            network_penalty = params['network_penalty']

        # Weight loss function based on training data label imbalance
        # see, e.g. https://discuss.pytorch.org/t/about-bcewithlogitslosss-pos-weights/22567/2
        #
        # TODO: could add a function argument to turn this on/off (but in
        # general it seems to give slightly better results)
        if classify:
            train_count = np.bincount(y_train)
            pos_weight = train_count[0] / train_count[1]
            if self.verbose:
                print('\n[0, 1]: {} (pos_weight={:.4f})'.format(train_count, pos_weight))

        if self.use_gpu:
            X_tr = torch.stack([torch.Tensor(x) for x in X_train]).cuda()
            X_ts = torch.stack([torch.Tensor(x) for x in X_test]).cuda()
            y_tr = torch.Tensor(y_train).view(-1, 1).cuda()
            y_ts = torch.Tensor(y_test).view(-1, 1).cuda()
            if classify:
                pos_weight = torch.Tensor([pos_weight]).cuda()
        else:
            X_tr = torch.stack([torch.Tensor(x) for x in X_train])
            X_ts = torch.stack([torch.Tensor(x) for x in X_test])
            y_tr = torch.Tensor(y_train).view(-1, 1)
            y_ts = torch.Tensor(y_test).view(-1, 1)
            if classify:
                pos_weight = torch.Tensor([pos_weight])

        train_loader = data_utils.DataLoader(
                data_utils.TensorDataset(X_tr, y_tr),
                batch_size=batch_size, shuffle=True)

        # TODO: this model should be polymorphic (classify/regress)
        model = LogisticRegression(X_train.shape[1])
        if self.use_gpu:
            model = model.cuda()

        # pos_weight is a scalar, the weight for the 1 class
        if classify:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #                 optimizer, patience=5)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                # add l1 loss
                # print([param for name, param in model.named_parameters()
                #              if 'bias' not in name][0])
                # weights = torch.Tensor(
                #         [param for name, param in model.named_parameters()
                #         if 'bias' not in name])
                # print(weights)

                l1_loss = sum(torch.norm(param, 1)
                                for name, param in model.named_parameters()
                                if 'bias' not in name)
                loss += l1_penalty * l1_loss

                # add network penalty if applicable
                if self.laplacian is not None:
                    w = [param for name, param in model.named_parameters()
                               if 'bias' not in name][0]
                    # filter for features that are in the network
                    w = w[:, self.network_features]
                    network_loss = torch.mm(
                        w.view(1, -1),
                        torch.sparse.mm(self.laplacian, w.view(-1, 1)))
                    loss += (network_penalty * network_loss).view(-1)[0]

                running_loss += loss
                loss.backward()
                optimizer.step()

            if learning_curves:
                # save train loss and test loss on whole dataset after each epoch
                y_pred_train = model(X_tr)
                y_pred_test = model(X_ts)
                self.monitor_['train_loss'].append(float((
                    criterion(y_pred_train, y_tr)
                ).detach()))
                self.monitor_['test_loss'].append(float((
                    criterion(y_pred_test, y_ts)
                ).detach()))
                # also save l1 loss
                self.monitor_['l1_loss'].append(float((
                    sum(torch.norm(param, 1)
                            for name, param in model.named_parameters()
                            if 'bias' not in name)
                ).detach()))
                # also save network loss
                network_weights = [param for name, param in model.named_parameters()
                                         if 'bias' not in name][0]
                network_weights = network_weights[:, self.network_features]
                network_loss = torch.mm(
                    w.view(1, -1),
                    torch.sparse.mm(self.laplacian, w.view(-1, 1)))
                self.monitor_['network_loss'].append(float((
                    (network_loss).view(-1)[0]
                ).detach()))
                network_weights = model.linear.weight.data.reshape(-1)

            # scheduler.step(running_loss)

        if save_weights:
            # bias goes first, then weights in order
            if self.use_gpu:
                bias = model.linear.bias.data.cpu().numpy()
                weights = model.linear.weight.data.cpu().numpy().flatten()
                self.last_weights = np.concatenate((bias, weights))
            else:
                bias = model.linear.bias.data.numpy()
                weights = model.linear.weight.data.numpy().flatten()
                self.last_weights = np.concatenate((bias, weights))

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

        if self.verbose:
            print('(time: {:.3f} sec)'.format(time.time() - t))

        if self.use_gpu:
            y_pred_train = y_pred_train.cpu().detach().numpy()
            y_pred_test = y_pred_test.cpu().detach().numpy()
        else:
            y_pred_train = y_pred_train.detach().numpy()
            y_pred_test = y_pred_test.detach().numpy()


        if classify:
            y_pred_bn_train = (y_pred_train > 0).astype('int')
            y_pred_bn_test = (y_pred_test > 0).astype('int')
        else:
            y_pred_bn_train = np.array([])
            y_pred_bn_test = np.array([])

        return ((train_loss, test_loss),
                (y_pred_train, y_pred_test),
                (y_pred_bn_train, y_pred_bn_test))


    def torch_param_selection(self, X_train, y_train):
        """Cross-validate to select best parameters from a set of possibilities.

        Dataset terminology: (subtrain | tune) = train | test
        (avoiding the term "validation" since it's overloaded in biology)
        """
        # k-fold cross-validation over the training data
        kf = KFold(n_splits=self.num_inner_folds, shuffle=True,
                   random_state=self.seed)
        results_df = None
        for fold, (subtrain_ixs, tune_ixs) in enumerate(kf.split(X_train), 1):
            X_subtrain, X_tune = X_train[subtrain_ixs], X_train[tune_ixs]
            y_subtrain, y_tune = y_train[subtrain_ixs], y_train[tune_ixs]
            if self.verbose:
                print('Running inner CV fold {} of {}'.format(
                        fold, self.num_inner_folds))
            result_df = self.torch_tuning(X_subtrain, X_tune, y_subtrain, y_tune)
            result_df['fold'] = fold
            if results_df is None:
                results_df = result_df
            else:
                results_df = pd.concat((results_df, result_df), ignore_index=True)

        # get the index of the parameter set that performed the best on
        # average across folds
        sorted_df = (
            results_df.loc[results_df['train/tune'] == 'tune']
                      .groupby('param_set')
                      .mean()
                      .sort_values(by='loss')
                      .reset_index()
        )
        best_ix = sorted_df.loc[0, 'param_set']
        best_params = {k: v[best_ix] for k, v in self.params_map.items()}

        # save CV results for best parameter set, for analysis later
        self.cv_results_df = results_df[results_df['param_set'] == best_ix]

        return results_df, best_params


    def torch_tuning(self, X_subtrain, X_tune, y_subtrain, y_tune):
        """Run parameter search on a single subtrain/tune split."""
        result = {
            'param_set': [],
            'train/tune': [],
            'loss': [],
            'auroc': []
        }
        for param in self.params_map.keys():
            result[param] = []
        num_iters = len(self.params_map[list(self.params_map.keys())[0]])
        for ix in range(num_iters):
            if self.verbose:
                print('-- Running parameter set {} of {}...'.format(ix+1, num_iters),
                      end='')
            params = {k: v[ix] for k, v in self.params_map.items()}
            losses, y_preds, __ = self.torch_model(X_subtrain,
                                                   X_tune,
                                                   y_subtrain,
                                                   y_tune,
                                                   params)
            y_pred_subtrain, y_pred_tune = y_preds
            subtrain_loss, tune_loss = losses
            subtrain_auroc = roc_auc_score(y_subtrain, y_pred_subtrain, average="weighted")
            tune_auroc = roc_auc_score(y_tune, y_pred_tune, average="weighted")
            if self.verbose:
                print('subtrain_loss: {:.4f}, tune_loss: {:.4f}'.format(
                        subtrain_loss, tune_loss))
            result['param_set'].append(ix)
            result['train/tune'].append('train')
            result['loss'].append(subtrain_loss)
            result['auroc'].append(subtrain_auroc)
            for param in self.params_map.keys():
                result[param].append(self.params_map[param][ix])
            result['param_set'].append(ix)
            result['train/tune'].append('tune')
            result['loss'].append(tune_loss)
            result['auroc'].append(tune_auroc)
            for param in self.params_map.keys():
                result[param].append(self.params_map[param][ix])
        return pd.DataFrame(result)


if __name__ == '__main__':
    # code to test the implementation quickly against sklearn
    # using breast cancer dataset from sklearn.datasets
    # original: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

    import argparse

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    import sys; sys.path.append('.')
    import config as cfg
    from tcga_util import get_threshold_metrics
    from classify_sklearn import train_sklearn_model

    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--l1_penalty', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=cfg.default_seed)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    # hyperparameter choices to do a random search over
    sklearn_param_choices = {
        'alpha': [0.1, 0.13, 0.15, 0.2, 0.25, 0.3],
        'l1_ratio': [0.15, 0.16, 0.2, 0.25, 0.3, 0.4]
    }

    """
    torch_param_choices = {
        'learning_rate': [0.001, 0.0001, 0.00001],
        'batch_size': [10, 20, 50, 100],
        'num_epochs': [100, 200, 500, 1000],
        'l1_penalty': [0, 0.01, 0.1, 1, 10]
    }
    """
    torch_param_choices = {
        'learning_rate': [0.0005],
        'batch_size': [50],
        'num_epochs': [200],
        'l1_penalty': [args.l1_penalty]
    }

    num_iters = 8
    num_inner_folds = 3
    model = TorchLR(torch_param_choices,
                    seed=args.seed,
                    num_iters=num_iters,
                    num_inner_folds=num_inner_folds,
                    use_gpu=args.gpu,
                    verbose=args.verbose)

    # load data and split into train/test sets
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=args.seed)

    # classify using sklearn SGDClassifier
    y_pred = train_sklearn_model(X_train, X_test,
                                 y_train,
                                 sklearn_param_choices['alpha'],
                                 sklearn_param_choices['l1_ratio'],
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

    losses, preds, preds_bn = model.train_torch_model(X_train, X_test,
                                                      y_train, y_test,
                                                      save_weights=True)

    y_pred_train, y_pred_test = preds
    y_pred_bn_train, y_pred_bn_test = preds_bn

    torch_train_acc = TorchLR.calculate_accuracy(y_train, y_pred_bn_train.flatten())
    torch_test_acc = TorchLR.calculate_accuracy(y_test, y_pred_bn_test.flatten())

    torch_train_results = get_threshold_metrics(y_train, y_pred_train)
    torch_test_results = get_threshold_metrics(y_test, y_pred_test)

    print('Sklearn train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        sk_train_acc, sk_test_acc))
    print('Sklearn train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        sk_train_results['auroc'], sk_test_results['auroc']))
    print('Sklearn train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        sk_train_results['aupr'], sk_test_results['aupr']))

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

    random_train_acc = TorchLR.calculate_accuracy(y_train, y_pred_bn_train)
    random_test_acc = TorchLR.calculate_accuracy(y_test, y_pred_bn_test)

    random_train_results = get_threshold_metrics(y_train, y_pred_train)
    random_test_results = get_threshold_metrics(y_test, y_pred_test)

    print('Random guessing train accuracy: {:.3f}, test accuracy: {:.3f}'.format(
        random_train_acc, random_test_acc))
    print('Random guessing train AUROC: {:.3f}, test AUROC: {:.3f}'.format(
        random_train_results['auroc'], random_test_results['auroc']))
    print('Random guessing train AUPRC: {:.3f}, test AUPRC: {:.3f}'.format(
        random_train_results['aupr'], random_test_results['aupr']))

