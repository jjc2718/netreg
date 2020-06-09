import os
import tempfile
import subprocess
import numpy as np
from collections import namedtuple

import config as cfg

def fit_gelnet_model(X_train, X_test, y_train, results_dir, seed,
                     verbose=True):

    # store data/labels in temp files to pass to R script
    train_data = tempfile.NamedTemporaryFile(mode='w', delete=False)
    test_data = tempfile.NamedTemporaryFile(mode='w', delete=False)
    train_labels = tempfile.NamedTemporaryFile(mode='w', delete=False)
    test_labels = tempfile.NamedTemporaryFile(mode='w', delete=False)
    np.savetxt(train_data, X_train.values, fmt='%.5f', delimiter='\t')
    np.savetxt(test_data, X_test.values, fmt='%.5f', delimiter='\t')
    np.savetxt(train_labels, y_train.status.values, fmt='%i')
    Filenames = namedtuple('Filenames', ['train_data', 'test_data',
                                         'train_labels', 'test_labels'])
    fnames = Filenames(
        train_data=train_data.name,
        test_data=test_data.name,
        train_labels=train_labels.name,
        test_labels=test_labels.name
    )
    train_data.close()
    test_data.close()
    train_labels.close()
    test_labels.close()

    # fit R model on training set and test on held out data
    r_args = [
        'Rscript',
        os.path.join(cfg.scripts_dir, 'run_gelnet.R'),
        '--train_data', fnames.train_data,
        '--test_data', fnames.test_data,
        '--train_labels', fnames.train_labels,
        '--results_dir', results_dir,
        '--seed', str(seed),
        '--ignore_network',
        '--classify',
        '--cv'
    ]
    if verbose:
        r_args.append('--verbose')
        print('Running: {}'.format(' '.join(r_args)))

    r_env = os.environ.copy()
    r_env['MKL_THREADING_LAYER'] = 'GNU'
    subprocess.check_call(r_args, env=r_env)
    # clean up temp files
    for fname in fnames:
        os.remove(fname)

    y_pred_train = np.loadtxt(
        os.path.join(results_dir,
                     'r_nn_preds_train_n0_p0_e0_u0_s{}.txt'.format(seed)),
        delimiter='\t')
    y_pred_test = np.loadtxt(
        os.path.join(results_dir,
                     'r_nn_preds_test_n0_p0_e0_u0_s{}.txt'.format(seed)),
        delimiter='\t')
    print(y_train[:10])
    print(y_pred_train[:10])
    exit()

    return (y_pred_train_df, y_pred_test_df, coef_df)
