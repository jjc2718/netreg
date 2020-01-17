import numpy as np

def save_numpy_to_el(adj, nodelist, filename):
    import networkx as nx
    G = nx.from_numpy_matrix(adj)
    G = nx.relabel_nodes(G, {ix: n for ix, n in enumerate(nodelist)})
    nx.write_weighted_edgelist(G, filename, delimiter='\t')

def generate_and_save_network(is_correlated, filename):
    import itertools as it

    n = is_correlated.shape[0]
    mnet = np.zeros((n, n))
    nodelist = np.arange(n)

    for (i, j) in it.combinations(range(n), 2):
        if is_correlated[i] and is_correlated[j]:
            mnet[i, j] = 1.0
            mnet[j, i] = 1.0

    for k in range(n):
        if not is_correlated[k]:
            mnet[k, k] = 1.0

    save_numpy_to_el(mnet, nodelist, filename)

def simulate_network(n, p, uncorr_frac, num_networks, seed=1, verbose=False):
    """Simulate data from a log-linear model with network collinearity."""
    import itertools as it
    import networkx as nx

    np.random.seed(seed)

    p_uncorr = int(uncorr_frac * p)
    p_corr = p - p_uncorr

    is_correlated = np.zeros(p).astype('bool')
    is_correlated[:p_corr] = True

    if verbose:
        print('Number of informative features: {}'.format(p_corr))
        print('Number of uninformative features: {}'.format(p_uncorr))

    adj_matrix = np.eye(p)
    network_groups = np.array_split(np.arange(p_corr), num_networks)
    for group in network_groups:
        for (i, j) in it.combinations(group, 2):
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    X = np.random.randn(n, p)
    B = np.zeros((p_corr,))
    for group in network_groups:
        B[group] = np.random.randn()
    B = np.concatenate((np.random.randn(1), B))

    # calculate Bernoulli parameter pi(x_i) for each sample x_i
    linsum = B[0] + (X[:, :p_corr] @ B[1:, np.newaxis])
    pis = 1 / (1 + np.exp(-linsum))

    y = np.random.binomial(1, pis.flatten())

    return (X, B, y, is_correlated, adj_matrix, network_groups)

def simulate_network_reg(n, p, uncorr_frac, num_networks, noise_stdev=0, seed=1, verbose=False):
    """Simulate data from a linear model with network collinearity."""
    import itertools as it
    import networkx as nx

    np.random.seed(seed)

    p_uncorr = int(uncorr_frac * p)
    p_corr = p - p_uncorr

    is_correlated = np.zeros(p).astype('bool')
    is_correlated[:p_corr] = True

    if verbose:
        print('Number of informative features: {}'.format(p_corr))
        print('Number of uninformative features: {}'.format(p_uncorr))

    adj_matrix = np.eye(p)
    network_groups = np.array_split(np.arange(p_corr), num_networks)
    for group in network_groups:
        for (i, j) in it.combinations(group, 2):
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    X = np.random.randn(n, p)
    B = np.zeros((p_corr,))
    for group in network_groups:
        B[group] = np.random.randn()
    B = np.concatenate((np.random.randn(1), B))

    # calculate Bernoulli parameter pi(x_i) for each sample x_i
    y = B[0] + (X[:, :p_corr] @ B[1:, np.newaxis])
    if noise_stdev > 0:
        y += np.random.normal(scale=noise_stdev, size=(n, 1))

    return (X, B, y, is_correlated, adj_matrix, network_groups)

if __name__ == '__main__':
    X, B, y, _, __, ___ = simulate_network_reg(100, 10, 0.2, 3, noise_stdev=1)
    print(X.shape)
    print(X[:10, :5])
    print(B.shape)
    print(B[:10])
    print(y.shape)
    print(y[:10, :])
