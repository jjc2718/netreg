import numpy as np

def shuffle_same(X, y):
    """Shuffle a data matrix and a label matrix in the same order."""
    assert X.shape[1] == y.shape[0]
    p = np.random.permutation(X.shape[1])
    return X[:, p], y[p]


def simulate_ll(n, p, uncorr_frac, seed=1, verbose=False):
    """Simulate data from a log-linear model.

    Parameters
    ----------
    n: int
        number of samples

    p: int
        number of features

    uncorr_frac: float
        Fraction of features to be uncorrelated with outcome (between 0 and 1)

    seed: int
        Seed for random number generator

    verbose: bool
        Whether to print verbose output or not

    """
    np.random.seed(seed)

    p_uncorr = int(uncorr_frac * p)
    p_corr = p - p_uncorr

    if verbose:
        print('Number of informative features: {}'.format(p_corr))
        print('Number of uninformative features: {}'.format(p_uncorr))

    # start by drawing features independently from a standard normal dist
    # TODO: could add an option for counts in the future?
    X_corr = np.random.randn(n, p_corr)
    X_uncorr = np.random.randn(n, p_uncorr)
    X = np.concatenate((X_corr, X_uncorr), axis=1)

    is_correlated = np.zeros(p).astype('bool')
    is_correlated[:p_corr] = True

    X, is_correlated = shuffle_same(X, is_correlated)

    # draw regression coefficients (betas) from standard normal
    B = np.random.randn(p_corr+1)

    # calculate Bernoulli parameter pi(x_i) for each sample x_i
    linsum = B[0] + (X_corr @ B[1:, np.newaxis])
    pis = 1 / (1 + np.exp(-linsum))

    y = np.random.binomial(1, pis.flatten())

    return (X, y, pis, is_correlated)


def split_train_test(n, train_frac, seed=1, verbose=False):
    np.random.seed(seed)

    n_train = int(train_frac * n)
    n_test = n - n_train

    if verbose:
        print('Train samples: {}, test samples: {}'.format(n_train, n_test))

    train_ixs = np.zeros(n).astype('bool')
    train_ixs[:n_train] = True
    np.random.shuffle(train_ixs)

    return train_ixs

