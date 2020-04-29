import tensorflow as tf

from gpflow.config import default_jitter
from gpflow.covariances.dispatch import Kuf, Kuu


def residual_variances(model):
    X_data, Y_data = model.data

    Kdiag = model.kernel(X_data, full_cov=False)
    kuu = Kuu(model.inducing_variable, model.kernel, jitter=default_jitter())
    kuf = Kuf(model.inducing_variable, model.kernel, X_data)

    L = tf.linalg.cholesky(kuu)
    A = tf.linalg.triangular_solve(L, kuf, lower=True)

    c = Kdiag - tf.reduce_sum(tf.square(A), 0)

    return c.numpy()
