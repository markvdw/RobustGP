import numpy as np
import scipy
from typing import Callable, Optional
from .methods import InducingPointInitializer


class RLS(InducingPointInitializer):
    """
    Implements a modified version of the "fixed size" variant of the (approximate)
    RLS algorithm given in Musco and Musco 2017
    @inproceedings{musco2017recursive,
                   title={Recursive sampling for the nystrom method},
                   author={Musco, Cameron and Musco, Christopher},
                   booktitle={Advances in Neural Information Processing Systems},
                   pages={3833--3845},
                   year={2017}
                   }
    """

    def __init__(self, seed: Optional[int] = 0, **kwargs):
        super().__init__(seed=seed, randomized=True, **kwargs)

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        indices, _, _ = recursive_rls(training_inputs, M, kernel, np.arange(training_inputs.shape[0]))
        return training_inputs[indices], indices


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
        indices = np.where(random_nums < probs)[0]
        # in cases where to few (potentially no) weights are sampled
        num_additional_indices = M - len(indices)
        if num_additional_indices > 0:
            candidate_indices = np.where(random_nums >= probs)[0]
            additional_indices = np.random.choice(candidate_indices, size=num_additional_indices,
                                                  replace=False)
            indices = np.append(indices, additional_indices)
        indices_to_include = active_indices[indices]
        column_weights = np.sqrt(1. / probs[indices])
    else:
        probs = probs * M / np.sum(probs)
        random_nums = np.random.rand(len(probs))
        indices_to_include = active_indices[random_nums < probs]
        column_weights = np.sqrt(1. / probs[random_nums < probs])
        # If we sample too few inducing points, resample
        while len(indices_to_include) < M:
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
    k = np.minimum(num_data, int(np.ceil(c * M / np.log(M+1))))

    if len(active_indices) <= M:  # Base case of recursion, see l 1,2 in Musco and Musco, alg 3
        return active_indices, np.ones_like(active_indices), np.ones_like(active_indices)
    s_bar = np.random.randint(0, 2, len(active_indices)).nonzero()[0]  # points sampled into Sbar, l4
    if len(s_bar) == 0:
        active_indices = np.random.choice(active_indices, (1+len(active_indices))//2, replace=False)
        return active_indices, np.ones_like(active_indices), np.ones_like(active_indices)

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