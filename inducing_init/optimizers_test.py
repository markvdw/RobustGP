import numpy as np
import pytest
import tensorflow as tf

import gpflow
from .models import RobustSGPR, RobustGPR
from .optimizers import RobustScipy
from .utilities import set_trainable

X = np.random.rand(500, 1)
Y = np.hstack((np.sin(X), np.cos(X)))


@pytest.mark.parametrize(
    "model",
    [
        RobustSGPR((X, Y), gpflow.kernels.SquaredExponential(), X.copy()),
        RobustGPR((X, Y), gpflow.kernels.SquaredExponential()),
    ],
)
def test_optimize_stability(model):
    model.jitter_variance.assign(1e-14)
    print(model.jitter_variance.numpy())
    model.likelihood.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive(lower=1e-16))
    set_trainable(model, False)
    set_trainable(model.kernel, True)
    set_trainable(model.likelihood, True)

    loss_function = model.training_loss_closure(compile=True)
    robust_loss_function = lambda: -model.robust_maximum_log_likelihood_objective()

    with pytest.raises(tf.errors.InvalidArgumentError):
        opt = gpflow.optimizers.Scipy()
        opt.minimize(loss_function, model.trainable_variables, method='l-bfgs-b',
                     options=dict(maxiter=10000))

    opt = RobustScipy()
    opt.minimize(loss_function, model.trainable_variables, robust_closure=robust_loss_function,
                 method='l-bfgs-b', options=dict(maxiter=10000))
    opt.minimize(loss_function, model.trainable_variables, robust_closure=robust_loss_function,
                 method='l-bfgs-b', options=dict(maxiter=10000))

    gpflow.utilities.print_summary(model)
