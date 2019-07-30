"""
Adapted from Tybalt data_models:
https://github.com/greenelab/tybalt/blob/master/tybalt/data_models.py

"""
import os
import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import config as cfg

class DataModel():
    """
    Methods for loading and compressing input data

    Usage:

    from data_models import DataModel
    data = DataModel(filename)

    """
    def __init__(self, filename=None, df=False, select_columns=False,
                 gene_modules=None, test_filename=None, test_df=None):
        """
        DataModel can be initialized with either a filename or a pandas
        dataframe and processes gene modules and sample labels if provided.
        Arguments:
        filename - if provided, load gene expression data into object
        df - dataframe of preloaded gene expression data
        select_columns - the columns of the dataframe to use
        gene_modules - a list of gene module assignments for each gene (for use
        with the simulated data or when ground truth gene modules are known)
        test_filename - if provided, loads testing dataset into object
        test_df - dataframe of prelaoded gene expression testing set data
        """
        # Load gene expression data
        self.filename = filename
        if filename is None:
            self.df = df
        else:
            self.df = pd.read_table(self.filename, index_col=0)

        if select_columns:
            subset_df = self.df.iloc[:, select_columns]
            other_columns = range(max(select_columns) + 1, self.df.shape[1])
            self.other_df = self.df.iloc[:, other_columns]
            self.df = subset_df
            if self.test_df is not None:
                self.test_df = self.test_df.iloc[:, select_columns]

        if gene_modules is not None:
            self.gene_modules = pd.DataFrame(gene_modules).T
            self.gene_modules.index = ['modules']

        self.num_samples, self.num_genes = self.df.shape

        # Load test set gene expression data if applicable
        self.test_filename = test_filename
        self.test_df = test_df

        if test_filename is not None and test_df is None:
            self.test_df = pd.read_table(self.test_filename, index_col=0)
            self.num_test_samples, self.num_test_genes = self.test_df.shape

            assert_ = 'train and test sets must have same number of genes'
            assert self.num_genes == self.num_test_genes, assert_


    def transform(self, how):
        self.transformation = how
        if how == 'zscore':
            self.transform_fit = StandardScaler().fit(self.df)
        elif how == 'zeroone':
            self.transform_fit = MinMaxScaler().fit(self.df)
        else:
            raise ValueError('how must be either "zscore" or "zeroone".')

        self.df = pd.DataFrame(self.transform_fit.transform(self.df),
                               index=self.df.index,
                               columns=self.df.columns)

        if self.test_df is not None:
            if how == 'zscore':
                self.transform_test_fit = StandardScaler().fit(self.test_df)
            elif how == 'zeroone':
                self.transform_test_fit = MinMaxScaler().fit(self.test_df)

            test_transform = self.transform_test_fit.transform(self.test_df)
            self.test_df = pd.DataFrame(test_transform,
                                        index=self.test_df.index,
                                        columns=self.test_df.columns)

    @classmethod
    def list_algorithms(self):
        return ['pca', 'sparse_pca', 'nmf', 'plier']

    def pca(self, n_components, transform_df=False, transform_test_df=False):
        self.pca_fit = decomposition.PCA(n_components=n_components)
        self.pca_df = self.pca_fit.fit_transform(self.df)
        colnames = ['pca_{}'.format(x) for x in range(0, n_components)]
        self.pca_df = pd.DataFrame(self.pca_df, index=self.df.index,
                                   columns=colnames)
        self.pca_weights = pd.DataFrame(self.pca_fit.components_,
                                        columns=self.df.columns,
                                        index=colnames)
        if transform_df:
            out_df = self.pca_fit.transform(self.df)
            return out_df

        if transform_test_df:
            self.pca_test_df = self.pca_fit.transform(self.test_df)


    def sparse_pca(self, n_components, transform_df=False,
                   transform_test_df=False,alpha=1, seed=1):
        self.spca_fit = decomposition.MiniBatchSparsePCA(n_components=n_components,
                                                         batch_size=50,
                                                         alpha=alpha,
                                                         random_state=seed,
                                                         normalize_components=True)
        self.spca_df = self.spca_fit.fit_transform(self.df)
        colnames = ['sparse_pca_{}'.format(x) for x in range(0, n_components)]
        self.spca_df = pd.DataFrame(self.spca_df, index=self.df.index,
                                    columns=colnames)
        self.spca_weights = pd.DataFrame(self.spca_fit.components_,
                                        columns=self.df.columns,
                                        index=colnames)
        if transform_df:
            out_df = self.spca_fit.transform(self.df)
            return out_df

        if transform_test_df:
            self.spca_test_df = self.spca_fit.transform(self.test_df)


    def nmf(self, n_components, transform_df=False, transform_test_df=False,
            seed=1, init='nndsvdar', tol=5e-3):
        self.nmf_fit = decomposition.NMF(n_components=n_components, init=init,
                                         tol=tol, random_state=seed)
        self.nmf_df = self.nmf_fit.fit_transform(self.df)
        colnames = ['nmf_{}'.format(x) for x in range(n_components)]

        self.nmf_df = pd.DataFrame(self.nmf_df, index=self.df.index,
                                   columns=colnames)
        self.nmf_weights = pd.DataFrame(self.nmf_fit.components_,
                                        columns=self.df.columns,
                                        index=colnames)
        if transform_df:
            out_df = self.nmf_fit.transform(self.df)
            return out_df

        if transform_test_df:
            self.nmf_test_df = self.nmf_fit.transform(self.test_df)


    def plier(self, n_components, pathways_file, transform_df=False,
              transform_test_df=False, shuffled=False, seed=1):
        import os
        import subprocess
        import tempfile
        plier_output_dir = os.path.join(cfg.data_dir, 'plier_output')
        if not os.path.exists(plier_output_dir):
            os.makedirs(plier_output_dir)
        output_prefix = os.path.join(plier_output_dir, 'plier_k{}_s{}'.format(
                                       n_components, seed))
        if shuffled:
            output_prefix += '_shuffled'
        output_data = output_prefix + '_z.tsv'
        output_weights = output_prefix + '_b.tsv'
        output_l2 = output_prefix + '_l2.tsv'

        if (not os.path.exists(output_data) or
            not os.path.exists(output_weights)):
            tf = tempfile.NamedTemporaryFile(mode='w')
            self.df.to_csv(tf, sep='\t')
            args = [
                'Rscript',
                os.path.join(cfg.scripts_dir, 'run_plier.R'),
                '--data', tf.name,
                '--k', str(n_components),
                '--seed', str(seed),
                '--pathways_file', pathways_file,
                '--output_prefix', output_prefix
            ]
            subprocess.check_call(args)
            tf.close()

        # The dimensions of matrices here are a bit confusing, since PLIER
        # does everything backward as compared to sklearn:
        #
        # - Input X has shape (n_features, n_samples)
        # - PLIER Z matrix has shape (n_features, n_components)
        # - PLIER B matrix has shape (n_components, n_samples)
        #
        # So in order to make this match the output of sklearn, set:
        #
        # - plier_df = PLIER B.T, has shape (n_samples, n_components)
        # - plier_weights = PLIER Z.T, has shape (n_components, n_features)
        self.plier_df = pd.read_csv(output_weights, sep='\t').T
        self.plier_weights = pd.read_csv(output_data, sep='\t').T
        plier_l2 = np.loadtxt(output_l2)

        # Filter to intersection of expression genes and genes in pathway
        # dataset (PLIER does this internally, but we also need to do it here
        # for the downstream analysis)
        test_df_filtered = self.test_df[self.plier_weights.columns.astype('str')]

        if transform_df:
            return self.plier_df
        if transform_test_df:
            self.plier_test_df = self._plier_on_test_data(test_df_filtered,
                                                          self.plier_weights,
                                                          plier_l2)

    def write_models(self, output_dir, file_suffix, test_set=False):
        """Write models (z matrices) to the given file.

        Arguments:
        output_dir - Directory to write models to
        file_suffix - Suffix of filename (containing, for example, the seed)
        """
        if hasattr(self, 'pca_df'):
            output_file = os.path.join(output_dir,
                                       'pca_{}'.format(file_suffix))
            if test_set:
                pca_df = pd.DataFrame(self.pca_test_df,
                                      index=self.test_df.index,
                                      columns=self.pca_df.columns)
            else:
                pca_df = self.pca_df
            pca_df.to_csv(output_file, sep='\t', compression='gzip')

        if hasattr(self, 'nmf_df'):
            output_file = os.path.join(output_dir,
                                       'nmf_{}'.format(file_suffix))
            if test_set:
                nmf_df = pd.DataFrame(self.nmf_test_df,
                                      index=self.test_df.index,
                                      columns=self.nmf_df.columns)
            else:
                nmf_df = self.nmf_df
            nmf_df.to_csv(output_file, sep='\t', compression='gzip')

        if hasattr(self, 'plier_df'):
            output_file = os.path.join(output_dir,
                                       'plier_{}'.format(file_suffix))
            if test_set:
                plier_df = pd.DataFrame(self.plier_test_df,
                                        index=self.test_df.index,
                                        columns=self.plier_df.columns)
            else:
                plier_df = self.plier_df
            plier_df.to_csv(output_file, sep='\t', compression='gzip')


    def write_weight_matrices(self, output_dir, file_suffix):
        """Write weight matrices to the given file.

        Arguments:
        output_dir - Directory to write models to
        file_suffix - Suffix of filename (containing, for example, the seed)
        """
        if hasattr(self, 'pca_df'):
            output_file = os.path.join(output_dir,
                                       'pca_{}'.format(file_suffix))
            self.pca_weights.to_csv(output_file, sep='\t', compression='gzip')
        if hasattr(self, 'nmf_df'):
            output_file = os.path.join(output_dir,
                                       'nmf_{}'.format(file_suffix))
            self.nmf_weights.to_csv(output_file, sep='\t', compression='gzip')
        if hasattr(self, 'plier_df'):
            output_file = os.path.join(output_dir,
                                       'plier_{}'.format(file_suffix))
            self.plier_weights.to_csv(output_file, sep='\t', compression='gzip')


    def compile_reconstruction(self, test_set=False):
        """
        Compile reconstruction costs between input and algorithm reconstruction
        Arguments:
        test_set - if True, compile reconstruction for the test set data
        Output:
        Two dictionaries storing 1) reconstruction costs and 2) reconstructed
        matrix for each algorithm
        """

        # Set the dataframe for use to compute reconstruction cost
        if test_set:
            input_df = self.test_df
        else:
            input_df = self.df

        all_reconstruction = {}
        reconstruct_mat = {}

        if hasattr(self, 'pca_df'):
            if test_set:
                pca_df = self.pca_test_df
            else:
                pca_df = self.pca_df

            pca_reconstruct = self.pca_fit.inverse_transform(pca_df)
            pca_recon = self._approx_keras_binary_cross_entropy(
                pca_reconstruct, input_df, self.num_genes)
            all_reconstruction['pca'] = [pca_recon]
            reconstruct_mat['pca'] = pd.DataFrame(pca_reconstruct,
                                                  index=input_df.index,
                                                  columns=input_df.columns)

        if hasattr(self, 'nmf_df'):
            if test_set:
                nmf_df = self.nmf_test_df
            else:
                nmf_df = self.nmf_df
            nmf_reconstruct = self.nmf_fit.inverse_transform(nmf_df)
            nmf_recon = self._approx_keras_binary_cross_entropy(
                nmf_reconstruct, input_df, self.num_genes)
            all_reconstruction['nmf'] = [nmf_recon]
            reconstruct_mat['nmf'] = pd.DataFrame(nmf_reconstruct,
                                                  index=input_df.index,
                                                  columns=input_df.columns)

        if hasattr(self, 'plier_df'):
            # have to do filtering to genes present in pathway dataset here too
            input_df = input_df[self.plier_weights.columns.astype('str')]
            num_genes = len(self.plier_weights.columns)
            if test_set:
                plier_df = self.plier_test_df
            else:
                plier_df = self.plier_df
            plier_reconstruct = np.dot(plier_df, self.plier_weights)
            plier_recon = self._approx_keras_binary_cross_entropy(
                plier_reconstruct, input_df, num_genes)
            all_reconstruction['plier'] = [plier_recon]
            reconstruct_mat['plier'] = pd.DataFrame(plier_reconstruct,
                                                    index=input_df.index,
                                                    columns=input_df.columns)

        return pd.DataFrame(all_reconstruction), reconstruct_mat

    def _plier_on_test_data(self, X, weights, lambda_2):
        """Apply PLIER latent space transformation to test data.

        This uses the sklearn ridge regression solver to find a representation
        of the test data in the latent space B that solves:

            argmin_B ||X - ZB||_Fro^2 + lambda_2 ||B||_Fro^2

        where X is the test data, and the transformation Z and the hyperparameter
        lambda_2 were fit by PLIER on the training data (and thus are constants
        here).

        Note that the other terms in the PLIER loss function are constant in B,
        so they can be ignored here.

        Parameters
        ----------
        X : array-like, [n_samples, n_features]
            Test gene expression data.

        weights : array-like, [n_components, n_features]
            Weights found by PLIER on training data.

        lambda_2 : float
            L2 parameter returned by PLIER, used as hyperparameter in RR.

        Returns
        -------
        array_like, [n_samples, n_components]
            Representation of the test data in the PLIER latent space.
        """
        from sklearn.linear_model import ridge_regression
        from scipy.stats import zscore
        return ridge_regression(weights.T,
                                X.apply(zscore).T,
                                lambda_2,
                                solver='svd')

    def _approx_keras_binary_cross_entropy(self, x, z, p, epsilon=1e-07):
        """
        Function to approximate Keras `binary_crossentropy()`
        https://github.com/keras-team/keras/blob/e6c3f77b0b10b0d76778109a40d6d3282f1cadd0/keras/losses.py#L76
        Which is a wrapper for TensorFlow `sigmoid_cross_entropy_with_logits()`
        https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        An important step is to clip values of reconstruction: see
        https://github.com/keras-team/keras/blob/a3d160b9467c99cbb27f9aa0382c759f45c8ee66/keras/backend/tensorflow_backend.py#L3071

        Parameters
        ----------
        x : array of float
            Reconstructed input RNAseq data

        z : array of float
            Input RNAseq data

        p : float
            number of features

        epsilon : float
            the clipping value to stabilize results (same Keras default)

        Returns
        -------
        float
            Approximation to the cross-entropy between x and z.

        """
        # Ensure numpy arrays
        x = np.array(x)
        z = np.array(z)

        # Add clip to value
        x[x < epsilon] = epsilon
        x[x > (1 - epsilon)] = (1 - epsilon)

        # Perform logit
        x = np.log(x / (1 - x))

        # Return approximate binary cross entropy
        return np.mean(p * np.mean(- x * z + np.log(1 + np.exp(x)), axis=-1))
