import numpy as np
import pytest
import tensorflow as tf

import gpflow
from .models import RobustSGPR, RobustGPR

np.random.seed(0)
X = np.random.rand(1000, 1)
Y = np.hstack((np.sin(X), np.cos(X)))


@pytest.mark.parametrize(
    "model",
    [
        RobustSGPR((X, Y), gpflow.kernels.SquaredExponential(), X.copy()),
        RobustGPR((X, Y), gpflow.kernels.SquaredExponential()),
    ],
)
def test_sgpr_stability(model):
    print(gpflow.config.default_jitter())

    # Setup hyperparmaeters
    initial_jitter = 1e-6
    model.kernel.variance.assign(2.3)
    model.kernel.lengthscales.assign(0.93)
    model.likelihood.variance.assign(1e-4)

    # For small jitter the results should be very close
    model.jitter_variance.assign(initial_jitter)
    nojitter = model.maximum_log_likelihood_objective()
    jitter = model.robust_maximum_log_likelihood_objective()
    np.testing.assert_allclose(jitter, nojitter)

    # Test that increasing jitter leads to a lower bound
    for j in np.logspace(1, 8, 8) * initial_jitter:
        model.jitter_variance.assign(initial_jitter * j)
        model.jitter_variance.assign(j)
        jitter = model.robust_maximum_log_likelihood_objective()
        print(nojitter.numpy(), jitter.numpy())
        assert jitter < nojitter

    # Test that adding jitter avoids a CholeskyError
    model.kernel.variance.assign(1e14)
    model.jitter_variance.assign(initial_jitter)

    with pytest.raises(tf.errors.InvalidArgumentError):
        model.maximum_log_likelihood_objective()

    model.robust_maximum_log_likelihood_objective()
    np.testing.assert_allclose(model.jitter_variance.numpy(), initial_jitter)
