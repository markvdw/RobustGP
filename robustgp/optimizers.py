from typing import Optional
from dataclasses import field
import numpy as np
import scipy
import tensorflow as tf

import gpflow
from gpflow.optimizers.scipy import (
    LossClosure,
    Variables,
    Tuple,
    _compute_loss_and_gradients,
    Callable,
    StepCallback,
    OptimizeResult,
)


class RobustScipy(gpflow.optimizers.Scipy):

    def __init__(self):
        super().__init__()
        self.f_vals = list()

    def minimize(
        self,
        closure: LossClosure,
        variables: Variables,
        method: Optional[str] = "L-BFGS-B",
        step_callback: Optional[StepCallback] = None,
        compile: bool = True,
        robust_closure: Optional[LossClosure] = None,
        **scipy_kwargs,
    ) -> OptimizeResult:
        """
        Minimize is a wrapper around the `scipy.optimize.minimize` function
        handling the packing and unpacking of a list of shaped variables on the
        TensorFlow side vs. the flat numpy array required on the Scipy side.

        Args:
            closure: A closure that re-evaluates the model, returning the loss
                to be minimized.
            variables: The list (tuple) of variables to be optimized
                (typically `model.trainable_variables`)
            method: The type of solver to use in SciPy. Defaults to "L-BFGS-B".
            step_callback: If not None, a callable that gets called once after
                each optimisation step. The callabe is passed the arguments
                `step`, `variables`, and `values`. `step` is the optimisation
                step counter. `variables` is the list of trainable variables as
                above, and `values` is the corresponding list of tensors of
                matching shape that contains their value at this optimisation
                step.
            compile: If True, wraps the evaluation function (the passed `closure` as
                well as its gradient computation) inside a `tf.function()`,
                which will improve optimization speed in most cases.

            scipy_kwargs: Arguments passed through to `scipy.optimize.minimize`

        Returns:
            The optimization result represented as a scipy ``OptimizeResult``
            object. See the Scipy documentation for description of attributes.
        """
        if not callable(closure):
            raise TypeError("The 'closure' argument is expected to be a callable object.")  # pragma: no cover
        variables = tuple(variables)
        if not all(isinstance(v, tf.Variable) for v in variables):
            raise TypeError(
                "The 'variables' argument is expected to only contain tf.Variable instances (use model.trainable_variables, not model.trainable_parameters)"
            )  # pragma: no cover
        initial_params = self.initial_parameters(variables)

        func = self.eval_func(closure, variables, compile=compile, robust_closure=robust_closure)
        if step_callback is not None:
            if "callback" in scipy_kwargs:
                raise ValueError("Callback passed both via `step_callback` and `callback`")

            callback = self.callback_func(variables, step_callback)
            scipy_kwargs.update(dict(callback=callback))

        return scipy.optimize.minimize(func, initial_params, jac=True, method=method, **scipy_kwargs)

    def eval_func(
        self,
        closure: LossClosure,
        variables: Variables,
        compile: bool = True,
        robust_closure: Optional[LossClosure] = None,
    ) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        def make_tf_eval(closure: LossClosure):
            def eager_tf_eval(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                values = self.unpack_tensors(variables, x)
                self.assign_tensors(variables, values)

                loss, grads = _compute_loss_and_gradients(closure, variables)
                return loss, self.pack_tensors(grads)

            return eager_tf_eval

        fast_tf_eval = make_tf_eval(closure)
        robust_tf_eval = make_tf_eval(robust_closure) if robust_closure is not None else None
        if compile:
            fast_tf_eval = tf.function(fast_tf_eval)  # Possibly compiled

        def _eval(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            try:
                loss, grad = fast_tf_eval(tf.convert_to_tensor(x))
            except tf.errors.InvalidArgumentError as e:
                e_msg = e.message
                if robust_tf_eval is None or (("Cholesky" not in e_msg) and ("not invertible" not in e_msg)):
                    raise e
                print(f"Warning: CholeskyError. Attempting to continue.")
                loss, grad = robust_tf_eval(tf.convert_to_tensor(x))
            self.f_vals.append(loss.numpy())
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        return _eval
