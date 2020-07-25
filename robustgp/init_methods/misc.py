import numpy as np
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
