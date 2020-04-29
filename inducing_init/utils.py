import numpy as np


def sample_discrete(unnormalized_probs):
    unnormalized_probs = np.clip(unnormalized_probs, 0, None)
    N = unnormalized_probs.shape[0]
    normalization = np.sum(unnormalized_probs)
    probs = unnormalized_probs / normalization
    return np.random.choice(a=N, size=1, p=probs)[0]
