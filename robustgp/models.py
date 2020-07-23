from typing import Optional

import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.config import default_jitter, default_float
from gpflow.covariances import Kuf, Kuu
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPR, SGPR
from gpflow.models.training_mixins import RegressionData, InputData
from gpflow.utilities import positive, to_default_float
from gpflow.models.model import MeanAndVariance


class RobustObjectiveMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter_variance = Parameter(
            max(default_jitter(), 1e-20), transform=positive(0.0), trainable=False, name="jitter"
        )

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
                    print(
                        f"{type(self).__name__}: Failed first computation. " f"Now attempting computation with jitter ",
                        end="",
                    )
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
                e_msg = e_inner.args
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

        # tr(Kff - Qff) should be positive, numerical issues can arise here
        assert trace_term > 0.0, f"Trace term negative, should be positive ({trace_term:.4e})."

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.math.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound -= trace_term

        return bound

    def upper_bound(self) -> tf.Tensor:
        """
        Upper bound for the sparse GP regression marginal likelihood.  Note that
        the same inducing points are used for calculating the upper bound, as are
        used for computing the likelihood approximation. This may not lead to the
        best upper bound. The upper bound can be tightened by optimising Z, just
        like the lower bound. This is especially important in FITC, as FITC is
        known to produce poor inducing point locations. An optimisable upper bound
        can be found in https://github.com/markvdw/gp_upper.

        The key reference is

        ::

          @misc{titsias_2014,
            title={Variational Inference for Gaussian and Determinantal Point Processes},
            url={http://www2.aueb.gr/users/mtitsias/papers/titsiasNipsVar14.pdf},
            publisher={Workshop on Advances in Variational Inference (NIPS 2014)},
            author={Titsias, Michalis K.},
            year={2014},
            month={Dec}
          }

        The key quantity, the trace term, can be computed via

        >>> _, v = conditionals.conditional(X, model.inducing_variable.Z, model.kernel,
        ...                                 np.zeros((len(model.inducing_variable), 1)))

        which computes each individual element of the trace term.
        """
        X_data, Y_data = self.data
        num_data = to_default_float(tf.shape(Y_data)[0])

        Kdiag = self.kernel(X_data, full_cov=False)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=self.jitter_variance)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)

        I = tf.eye(tf.shape(kuu)[0], dtype=default_float())

        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = I + AAT / self.likelihood.variance
        LB = tf.linalg.cholesky(B)

        # Using the Trace bound, from Titsias' presentation
        c = tf.maximum(tf.reduce_sum(Kdiag) - tf.reduce_sum(tf.square(A)), 0)

        # Alternative bound on max eigenval:
        corrected_noise = self.likelihood.variance + c

        const = -0.5 * num_data * tf.math.log(2 * np.pi * self.likelihood.variance)
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        LC = tf.linalg.cholesky(I + AAT / corrected_noise)
        v = tf.linalg.triangular_solve(LC, tf.linalg.matmul(A, Y_data) / corrected_noise, lower=True)
        quad = -0.5 * tf.reduce_sum(tf.square(Y_data)) / corrected_noise + 0.5 * tf.reduce_sum(tf.square(v))

        return const + logdet + quad

    def upper_bound(self) -> tf.Tensor:
        """
        Upper bound for the sparse GP regression marginal likelihood.  Note that
        the same inducing points are used for calculating the upper bound, as are
        used for computing the likelihood approximation. This may not lead to the
        best upper bound. The upper bound can be tightened by optimising Z, just
        like the lower bound. This is especially important in FITC, as FITC is
        known to produce poor inducing point locations. An optimisable upper bound
        can be found in https://github.com/markvdw/gp_upper.
        The key reference is
        ::
          @misc{titsias_2014,
            title={Variational Inference for Gaussian and Determinantal Point Processes},
            url={http://www2.aueb.gr/users/mtitsias/papers/titsiasNipsVar14.pdf},
            publisher={Workshop on Advances in Variational Inference (NIPS 2014)},
            author={Titsias, Michalis K.},
            year={2014},
            month={Dec}
          }
        The key quantity, the trace term, can be computed via
        >>> _, v = conditionals.conditional(X, model.inducing_variable.Z, model.kernel,
        ...                                 np.zeros((len(model.inducing_variable), 1)))
        which computes each individual element of the trace term.
        """
        X_data, Y_data = self.data
        num_data = to_default_float(tf.shape(Y_data)[0])

        Kdiag = self.kernel(X_data, full_cov=False)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=self.jitter_variance)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)

        I = tf.eye(tf.shape(kuu)[0], dtype=default_float())

        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = I + AAT / self.likelihood.variance
        LB = tf.linalg.cholesky(B)

        # Using the Trace bound, from Titsias' presentation
        c = tf.maximum(tf.reduce_sum(Kdiag) - tf.reduce_sum(tf.square(A)), 0)

        # Alternative bound on max eigenval:
        corrected_noise = self.likelihood.variance + c

        const = -0.5 * num_data * tf.math.log(2 * np.pi * self.likelihood.variance)
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        LC = tf.linalg.cholesky(I + AAT / corrected_noise)
        v = tf.linalg.triangular_solve(LC, tf.linalg.matmul(A, Y_data) / corrected_noise, lower=True)
        quad = -0.5 * tf.reduce_sum(tf.square(Y_data)) / corrected_noise + 0.5 * tf.reduce_sum(tf.square(v))

        return const + logdet + quad

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        X_data, Y_data = self.data
        num_inducing = len(self.inducing_variable)
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=self.jitter_variance)
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])
        return mean + self.mean_function(Xnew), var


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
            lambda: (
                tf.linalg.cholesky(tf.linalg.set_diag(K, k_diag + self.likelihood.variance)),
                tf.linalg.cholesky(tf.linalg.set_diag(K, k_diag + self.jitter_variance)),
            ),
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
