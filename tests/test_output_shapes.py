import os
import pytest
import tempfile
import numpy as np
import pandas as pd

import sys; sys.path.append('.')
import config as cfg
from data_models import DataModel

@pytest.fixture
def shapes_test():
    # n = number of samples
    # p = number of features
    # k = latent space dimension
    # m = number of pathways for PLIER
    np.random.seed(cfg.default_seed)
    params = {
        'n_train': 10,
        'n_test': 5,
        'p': 20,
        'k': 5,
        'm': 10
    }
    exp_data = {
        'train': pd.DataFrame(
            np.random.uniform(size=(params['n_train'], params['p'])),
            index=['S{}'.format(i) for i in range(params['n_train'])],
            columns=['G{}'.format(j) for j in range(params['p'])]),
        'test': pd.DataFrame(
            np.random.uniform(size=(params['n_test'], params['p'])),
            index=['S{}'.format(i) for i in range(
                        params['n_train'], params['n_train']+params['n_test'])],
            columns=['G{}'.format(j) for j in range(params['p'])])
    }
    return params, exp_data


def _generate_and_save_pathways(p, m):
    """Function to generate random pathways and save in a temporary file.

    File is closed but not deleted, so code that calls this function
    must delete the file after it is used.
    """
    # it really doesn't matter exactly what the simulated pathways are,
    # this is just used for testing that dimensions are correct
    pathways = pd.DataFrame(np.random.randint(2, size=(p, m)),
                            index=['G{}'.format(j) for j in range(p)],
                            columns=['PW{}'.format(k) for k in range(m)])
    tf = tempfile.NamedTemporaryFile(mode='w', delete=False)
    filename = tf.name
    pathways.to_csv(filename, sep='\t')
    tf.close()
    return filename


def test_pca_output(shapes_test):
    """Test dimensions of PCA output."""
    params, exp_data = shapes_test
    dm = DataModel(df=exp_data['train'], test_df=exp_data['test'])
    dm.transform(how='zscore')
    dm.pca(n_components=params['k'], transform_test_df=True)
    assert dm.pca_df.shape == (params['n_train'], params['k'])
    assert dm.pca_test_df.shape == (params['n_test'], params['k'])
    assert dm.pca_weights.shape == (params['k'], params['p'])


def test_plier_output(shapes_test):
    """Test dimensions of PLIER output."""
    params, exp_data = shapes_test
    pathways_file = _generate_and_save_pathways(params['p'],
                                                params['m'])
    dm = DataModel(df=exp_data['train'], test_df=exp_data['test'])
    dm.transform(how='zscore')
    dm.plier(n_components=params['k'], pathways_file=pathways_file,
             transform_test_df=True, skip_cache=True)
    os.remove(pathways_file)
    assert dm.plier_df.shape == (params['n_train'], params['k'])
    assert dm.plier_test_df.shape == (params['n_test'], params['k'])
    assert dm.plier_weights.shape == (params['k'], params['p'])

