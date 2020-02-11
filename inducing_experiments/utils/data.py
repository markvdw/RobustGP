from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from shutil import copyfile, rmtree

import numpy as np
from bayesian_benchmarks import data as bb_data
from observations.util import maybe_download_and_extract
from bayesian_benchmarks.data import *


def snelson1d(path):
    """Load Edward Snelson's 1d regression data set [@snelson2006fitc].
    It contains 200 examples of a few oscillations of an example function. It has
    seen extensive use as a toy dataset for illustrating qualitative behaviour of
    Gaussian process approximations.

    Args:
      path: str.
        Path to directory which either stores file or otherwise file will be
        downloaded and extracted there. Filenames are `snelson_train_*`.

    Returns:
      Tuple of two np.darray `inputs` and `outputs` with 200 rows and 1 column.
    """
    path = os.path.expanduser(path)
    inputs_path = os.path.join(path, 'snelson_train_inputs')
    outputs_path = os.path.join(path, 'snelson_train_outputs')

    # Contains all source as well. We just need the data.
    url = 'http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip'

    if not (os.path.exists(inputs_path) and os.path.exists(outputs_path)):
        maybe_download_and_extract(path, url)

        # Copy the required data
        copyfile(os.path.join(path, "SPGP_dist", "train_inputs"), inputs_path)
        copyfile(os.path.join(path, "SPGP_dist", "train_outputs"), outputs_path)

        # Clean up everything else
        rmtree(os.path.join(path, "SPGP_dist"))
        os.remove(os.path.join(path, "SPGP_dist.zip"))

    X = np.loadtxt(os.path.join(inputs_path))[:, None]
    Y = np.loadtxt(os.path.join(outputs_path))[:, None]

    return X, Y


def kin40k(dtype=np.float64):
    data = bb_data.Wilson_kin40k(split=1, prop=0.8)
    X = data.X_train.astype(dtype)
    Y = data.Y_train.astype(dtype)
    Xt = data.X_test.astype(dtype)
    Yt = data.Y_test.astype(dtype)
    return (X, Y), (Xt, Yt)
