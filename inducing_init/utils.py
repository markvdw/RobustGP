from typing import Callable, Optional

import numpy as np
import scipy.linalg
import warnings
np.seterr(all='raise')


def sample_discrete(unnormalized_probs):
    unnormalized_probs = np.clip(unnormalized_probs, 0, None)
    N = unnormalized_probs.shape[0]
    normalization = np.sum(unnormalized_probs)
    if normalization == 0:  # if all of the probabilities are numerically 0, sample uniformly
        warnings.warn("Trying to sample discrete distribution with all 0 weights")
        return np.random.choice(a=N, size=1)[0]
    probs = unnormalized_probs / normalization
    return np.random.choice(a=N, size=1, p=probs)[0]


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


def approximate_rls(training_inputs, kernel, regularization, subset_to_predict, subset_used, column_weights):
    X = training_inputs[subset_to_predict]
    Z = training_inputs[subset_used]
    regularization_matrix = np.diag(np.square(1. / column_weights) * regularization)
    regularized_Kuu = kernel(Z) + regularization_matrix
    L = np.linalg.cholesky(regularized_Kuu)
    kuf = kernel(Z, X)
    Linvkuf = scipy.linalg.solve_triangular(L, kuf, lower=True)
    posterior_variance = kernel(X, full_cov=False) - np.sum(np.square(Linvkuf), axis=0)

    return 1 / regularization * posterior_variance


def get_indices_and_weights(weighted_leverage, active_indices, k, top_level, M):
    probs = np.minimum(1., weighted_leverage * np.log(2*k))
    if not top_level:
        random_nums = np.random.rand(len(probs))
        indices_to_include = active_indices[random_nums < probs]
        column_weights = np.sqrt(1. / probs[random_nums < probs])
    else:
        probs = probs * M / np.sum(probs)
        random_nums = np.random.rand(len(probs))
        indices_to_include = active_indices[random_nums < probs]
        column_weights = np.sqrt(1. / probs[random_nums < probs])
        # If we sample too few inducing points, resample
        while len(indices_to_include) < M:
            print("Resampling, Sampled:", len(indices_to_include), "Target M:", M)
            random_nums = np.random.rand(len(probs))  # resample if not enough
            indices_to_include = active_indices[random_nums < probs]
            column_weights = np.sqrt(1. / probs[random_nums < probs])
            probs = np.clip(probs * M / np.sum(np.clip(probs, 0, 1)), 0, 1)
            probs *= 1.01
        inds = np.random.choice(len(indices_to_include), size=M, replace=False)
        indices_to_include, column_weights = indices_to_include[inds], column_weights[inds]
    return indices_to_include, column_weights, probs

def recursive_rls(training_inputs: np.ndarray,
                  M: int,
                  kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
                  active_indices: np.ndarray):
    num_data = training_inputs.shape[0]
    top_level = len(active_indices) == num_data   # boolean indicating we are at top level of recursion
    c = .25
    k = np.minimum(num_data, int(np.ceil(c * M / np.log(M))))


    if len(active_indices) <= M:  # Base case of recursion, see l 1,2 in Musco and Musco, alg 3
        return active_indices, np.ones_like(active_indices), np.ones_like(active_indices)
    s_bar = np.random.randint(0, 2, len(active_indices)).nonzero()[0]  # points sampled into Sbar, l4
    indices_to_include, column_weights, probs = recursive_rls(training_inputs, M, kernel,
                                                              active_indices=active_indices[s_bar])
    Z = training_inputs[indices_to_include]
    SKS = kernel(Z) * column_weights[None, :] * column_weights[:, None]  # sketched kernel matrix
    eigvals = scipy.sparse.linalg.eigsh(SKS.numpy(), k=k, which='LM', return_eigenvectors=False)
    lam = 1 / k * (np.sum(np.diag(SKS)) - np.sum(eigvals))
    lam = np.maximum(1e-12, lam)

    weighted_leverage = approximate_rls(training_inputs, kernel, lam, subset_to_predict=active_indices,
                                        subset_used=indices_to_include, column_weights=column_weights)
    indices_to_include, column_weights, probs = get_indices_and_weights(weighted_leverage, active_indices, k,
                                                                        top_level, M)

    return indices_to_include, column_weights, probs

