import numpy as np
import pytest

from gpflow.kernels import SquaredExponential
from .methods import FirstSubsample, UniformSubsample, Kmeans, ConditionalVariance


@pytest.mark.parametrize("init_method", [FirstSubsample(seed=0), UniformSubsample(seed=0), Kmeans(seed=0),
                                         ConditionalVariance(seed=0, sample=True),
                                         ConditionalVariance(seed=0, sample=False)])
def test_seed_reproducibility(init_method):
    k = SquaredExponential()
    X = np.random.randn(100, 2)

    Z1, idx1 = init_method(X, 30, k)
    Z2, idx2 = init_method(X, 30, k)

    assert np.all(Z1 == Z2), str(init_method)
    assert np.all(idx1 == idx2), str(init_method)


def test_incremental_ConditionalVariance():
    init_method = ConditionalVariance(sample=True)

    k = SquaredExponential()
    X = np.random.randn(100, 2)

    Z1, idx1 = init_method(X, 20, k)
    Z2, idx2 = init_method(X, 30, k)

    assert np.all(Z1 == Z2[:20])
    assert np.all(idx1 == idx2[:20])
