import os
import unittest
import pytest
import tempfile
import numpy as np
import pandas as pd

import sys; sys.path.append('.')
import config as cfg
from data_models import DataModel

class ShapesTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ShapesTest, self).__init__(*args, **kwargs)
        # n = number of samples
        # p = number of features
        # k = latent space dimension
        # m = number of pathways for PLIER
        self.n_train = 10
        self.n_test = 5
        self.p = 20
        self.k = 5
        self.m = 10
        np.random.seed(cfg.default_seed)
        self.exp_data_train = pd.DataFrame(
                                np.random.uniform(size=(self.n_train, self.p)),
                                index=['S{}'.format(i) for i in range(self.n_train)],
                                columns=['G{}'.format(j) for j in range(self.p)])
        self.exp_data_test = pd.DataFrame(
                                np.random.uniform(size=(self.n_test, self.p)),
                                index=['S{}'.format(i)
                                        for i in range(self.n_train,
                                                       self.n_train+self.n_test)],
                                columns=['G{}'.format(j) for j in range(self.p)])


    def test_pca_output(self):
        """Test dimensions of PCA output."""
        n_train, n_test, p, k = self.n_train, self.n_test, self.p, self.k
        dm = DataModel(df=self.exp_data_train, test_df=self.exp_data_test)
        dm.transform(how='zscore')
        dm.pca(n_components=k, transform_test_df=True)
        assert dm.pca_df.shape == (n_train, k)
        assert dm.pca_test_df.shape == (n_test, k)
        assert dm.pca_weights.shape == (k, p)


    def _generate_and_save_pathways(self, p, m):
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


    def test_plier_output(self):
        """Test dimensions of PLIER output."""
        n_train, n_test, p, k, m = self.n_train, self.n_test, self.p, self.k, self.m
        pathways_file = self._generate_and_save_pathways(p, m)
        dm = DataModel(df=self.exp_data_train, test_df=self.exp_data_test)
        dm.transform(how='zscore')
        dm.plier(n_components=k, pathways_file=pathways_file,
                 transform_test_df=True, skip_cache=True)
        os.remove(pathways_file)
        assert dm.plier_df.shape == (n_train, k)
        assert dm.plier_test_df.shape == (n_test, k)
        assert dm.plier_weights.shape == (k, p)

