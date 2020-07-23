import numpy as np
import scipy
import warnings
from typing import Callable, Optional

from .methods import ConditionalVariance


class KdppMCMC(ConditionalVariance):

    def __init__(self, num_steps: Optional[int] = 10000, seed: Optional[int] = 0, **kwargs):
        """
        Implements the MCMC approximation to sampling from a k-DPP developed in
        @inproceedings{anari2016monte,
                       title={Monte Carlo Markov chain algorithms for sampling strongly Rayleigh distributions and determinantal point processes},
                       author={Anari, Nima and Gharan, Shayan Oveis and Rezaei, Alireza},
                       booktitle={Conference on Learning Theory},
                       pages={103--115},
                       year={2016}
                    }
        and used for initializing inducing point in
        @inproceedings{burt2019rates,
                       title={Rates of Convergence for Sparse Variational Gaussian Process Regression},
                       author={Burt, David and Rasmussen, Carl Edward and Van Der Wilk, Mark},
                       booktitle={International Conference on Machine Learning},
                       pages={862--871},
                      year={2019}
            }
        More information on determinantal point processes and related algorithms can be found at:
        https://github.com/guilgautier/DPPy
        :param sample: int, number of steps of MCMC to run
        :param threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx.
        has converged.
        """
        super().__init__(seed=seed, **kwargs)
        self.num_steps = num_steps

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        """
        :param training_inputs: training_inputs: [N,D] numpy array
        :param M: int, number of inducing inputs to return
        :param kernel: kernelwrapper object
        :param num_steps: number of swap steps to perform.
        :param init_indices: array of M indices or None, set used to initialize mcmc alg. if None, we use the greedy MAP
        init. (variance, with sample=False)
        :return: inducing inputs, indices, [M], np.array of ints indices of these inputs in training data array
        """
        N = training_inputs.shape[0]
        _, indices = super().compute_initialisation(training_inputs, M, kernel)
        kzz = kernel(training_inputs[indices], None, full_cov=True)
        Q, R = scipy.linalg.qr(kzz, overwrite_a=True)
        if np.min(np.abs(np.diag(R))) == 0:
            warnings.warn("Determinant At initialization is numerically 0, MCMC was not run")
            return training_inputs[indices], indices
        for _ in range(self.num_steps):
            if np.random.rand() < .5:  # lazy MCMC, half the time, no swap is performed
                continue
            indices_complement = np.delete(np.arange(N), indices)
            s = np.random.randint(M)
            t = np.random.choice(indices_complement)
            swap, Q, R = accept_or_reject(training_inputs, kernel, indices, s, t, Q, R)
            if swap:
                indices = np.delete(indices, s, axis=0)
                indices = np.append(indices, [t], axis=0)
        return training_inputs[indices], indices


def _delete_qr_square(Q, R, s):
    """
    Given a QR decomposition of a square matrix K, remove row and column s. Note we do not overwrite the original
    matrices
    :param Q: orthogonal matrix from QR decomposition of K
    :param R: upper triangular matrix. K = QR
    :param s: index of row/column to remove
    :return: QR decomposition of matrix formed by deleting row/column s from K
    """
    # remove the corresponding row and column from QR-decomposition
    Qtemp, Rtemp = scipy.linalg.qr_delete(Q, R, s, which='row')
    Qtemp, Rtemp = scipy.linalg.qr_delete(Qtemp, Rtemp, s, which='col', overwrite_qr=True)
    return Qtemp, Rtemp


def _build_new_row_column(X, kernel, indices, s, t):
    ZminusS = np.delete(X[indices], s, axis=0)
    xt = X[t:t + 1]
    ktt = kernel(xt, None, full_cov=True)[0]
    row_to_add = kernel(ZminusS, xt)[:, 0]
    col_to_add = np.concatenate([row_to_add, ktt], axis=0)
    return row_to_add, col_to_add


def _add_qr_square(Q, R, row, col):
    # get the shape, Note that QR is square. We insert the column and row in the last position
    M = Q.shape[0]
    # Add the row and column to the matrix
    Qtemp, Rtemp = scipy.linalg.qr_insert(Q, R, row, M, which='row', overwrite_qru=True)
    Qtemp, Rtemp = scipy.linalg.qr_insert(Qtemp, Rtemp, col, M, which='col', overwrite_qru=True)
    return Qtemp, Rtemp


def _get_log_det_ratio(X, kernel, indices, s, t, Q, R):
    """
    Returns the log determinant ratio, as well as the updates to Q and R if the point is swapped
    :param X: training inputs
    :param kernel: kernelwrapper
    :param indices: X[indices]=Z
    :param s: current point we might (X[indices])[s] is the current point we might swap out
    :param t: X[t] is the point we might add
    :param Q: orthogonal matrix
    :param R: upper triangular matrix, QR = Kuu (for Z=X[indices])
    :return: log determinant ratio , Qnew, Rnew. QR decomposition of Z-{s}+{t}.
    """
    log_denominator = np.sum(np.log(np.abs(np.diag(R))))  # current value of det(Kuu)
    # remove s from the QR decomposition
    Qtemp, Rtemp = _delete_qr_square(Q, R, s)
    # build new row and column to add to QR decomposition
    row_to_add, col_to_add = _build_new_row_column(X, kernel, indices, s, t)
    # add the corresponding row and column to QR-decomposition
    Qnew, Rnew = _add_qr_square(Qtemp, Rtemp, row_to_add, col_to_add)
    # sometimes Rnew will have zero entries along the diagonal for numerical reasons, in these case, we should always
    # reject the swap as one Kuu should not be able to have (near) zero determinant, we force numpy to raise an error
    # and catch it
    try:
        log_numerator = np.sum(np.log(np.abs(np.diag(Rnew))))  # det(Kuu) if we perform swap
    except FloatingPointError:
        return - np.inf, Qnew, Rnew
    log_det_ratio = log_numerator - log_denominator  # ratio of determinants
    return log_det_ratio, Qnew, Rnew


def accept_or_reject(X, kernel, indices, s, t, Q, R):
    """
    Decides whether or not to swap points. Updates QR accordingly. Seems reasonably stable. Error of QR will get big if
    10k or more iterations are run (e.g. increases by about a factor of 10 over 10k
    iterations). Consider recomputing occasionally.
    :param X: candidate points
    :param k: kernel
    :param indices: Current active set (Z)
    :param s: index of point that could be removed in Z (Note this point is (X[indices])[s], not X[s]!)
    :param t: index of point that could be added in X
    :return: swap, Q, R: bool, [M,M], [M,M]. If swap, we removed s and added t. Return updated Q and R accodingly.
    """
    log_det_ratio, Qnew, Rnew = _get_log_det_ratio(X, kernel, indices, s, t, Q, R)
    acceptance_prob = np.exp(log_det_ratio)  # P(new)/P(old), probability of swap
    if np.random.rand() < acceptance_prob:
        return True, Qnew, Rnew  # swapped
    return False, Q, R  # stayed in same state