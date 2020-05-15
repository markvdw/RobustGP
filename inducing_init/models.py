from typing import Optional

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.config import default_jitter, default_float
from gpflow.covariances import Kuf, Kuu
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPR, SGPR
from gpflow.models.training_mixins import RegressionData
from gpflow.utilities import positive, to_default_float


class RobustObjectiveMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter_variance = Parameter(default_jitter(), transform=positive(0.0), trainable=False, name="jitter")

    def _compute_robust_maximum_log_likelihood_objective(self) -> tf.Tensor:
        raise NotImplementedError

    def robust_maximum_log_likelihood_objective(self, restore_jitter=True) -> tf.Tensor:
        initial_jitter = self.jitter_variance.numpy()
        N_orders = 20
        for i in range(N_orders):
            self.jitter_variance.assign(10 ** i * initial_jitter)
            logjitter = np.log10(self.jitter_variance.numpy())
            if i > 0:
                if i == 1:
                    print(f"{type(self).__name__}: Failed first computation. "
                          f"Now attempting computation with jitter ", end="")
                print(f"10**{logjitter:.2f} ", end="", flush=True)
            try:
                val = self._compute_robust_maximum_log_likelihood_objective()
                break
            except tf.errors.InvalidArgumentError as e_inner:
                e_msg = e_inner.message
                if (("Cholesky" not in e_msg) and ("not invertible" not in e_msg)) or i == (N_orders - 1):
                    print(e_msg)
                    raise e_inner
            except AssertionError as e_inner:
                e_msg = e_inner.message
                if i == (N_orders - 1):
                    print(e_msg)
                    raise e_inner
        if restore_jitter:
            self.jitter_variance.assign(initial_jitter)
        if i > 0:
            print("")
        return val


class RobustSGPR(RobustObjectiveMixin, SGPR):
    def _compute_robust_maximum_log_likelihood_objective(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        X_data, Y_data = self.data

        num_inducing = len(self.inducing_variable)
        num_data = to_default_float(tf.shape(Y_data)[0])
        output_dim = to_default_float(tf.shape(Y_data)[1])

        err = Y_data - self.mean_function(X_data)
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=self.jitter_variance)
        L = tf.linalg.cholesky(kuu)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        trace_term = 0.5 * output_dim * tf.reduce_sum(Kdiag) / self.likelihood.variance
        trace_term -= 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))
        assert trace_term > 0. # tr(Kff - Qff) should be positive, numerical issues can arise here

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.math.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound -= trace_term


        return bound


class RobustGPR(RobustObjectiveMixin, GPR):
    def __init__(
            self,
            data: RegressionData,
            kernel: Kernel,
            mean_function: Optional[MeanFunction] = None,
            noise_variance: float = 1.0,
    ):
        super().__init__(data, kernel, mean_function, noise_variance)

    def _compute_robust_maximum_log_likelihood_objective(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood, with some slack caused by the
        jitter. Adding the jitter ensures numerical stability.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        num_data = X.shape[0]
        output_dim = tf.shape(Y)[1]

        K = self.kernel(X)
        k_diag = tf.linalg.diag_part(K)
        noiseK_L, L = tf.cond(
            self.likelihood.variance > self.jitter_variance,
            lambda: (tf.linalg.cholesky(tf.linalg.set_diag(K, k_diag + self.likelihood.variance)),
                     tf.linalg.cholesky(tf.linalg.set_diag(K, k_diag + self.jitter_variance))),
            lambda: (tf.linalg.cholesky(tf.linalg.set_diag(K, k_diag + self.jitter_variance)),) * 2,
        )

        err = Y - self.mean_function(X)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, K, lower=True) / sigma

        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = tf.linalg.set_diag(AAT, tf.linalg.diag_part(AAT) + 1)  # B = AAT + tf.eye(num_data, dtype=default_float())
        # B = AAT + tf.eye(num_data, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * to_default_float(num_data) * to_default_float(output_dim) * np.log(2 * np.pi)
        bound -= to_default_float(output_dim) * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(noiseK_L)))
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))

        return bound
