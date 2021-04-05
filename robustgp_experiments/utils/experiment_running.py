import datetime
from dataclasses import dataclass, field, _MISSING_TYPE
from functools import reduce
from glob import glob
from typing import Optional

import gpflow
import json_tricks
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from robustgp import InducingPointInitializer, FirstSubsample
from robustgp.models import RobustGPR, RobustSGPR
from robustgp.optimizers import RobustScipy
from robustgp.utilities import set_trainable
from . import data
from .storing import get_next_filename


def create_loss_function(model, data):
    @tf.function(autograph=False)
    def loss():
        return -model.log_likelihood(*data)

    return loss


class LoggerCallback:
    def __init__(self, model, loss_function, holdout_interval=10):
        self.model = model
        self.loss_function = loss_function
        self.holdout_interval = holdout_interval
        self.log_likelihoods = []
        self.n_iters = []
        self.counter = 0

    def __call__(self, step, variables=None, values=None):
        # step will reset to zero between calls to minimize(), whereas counter will keep increasing
        if (self.counter <= 10) or (self.counter % self.holdout_interval) == 0:
            if variables is not None:
                # Using Scipy and need to update the parameters
                for var, val in zip(variables, values):
                    var.assign(val)

            self.n_iters.append(self.counter + 1)
            loss = self.loss_function().numpy()
            self.log_likelihoods.append(loss)
            print(f"{self.counter} - objective function: {self.log_likelihoods[-1]:.4f}", end="\r")

        self.counter += 1


# def run_tf_optimizer(model, optimizer, data, iterations, callback=None):
#     logf = []
#
#     @tf.function(autograph=False)
#     def optimization_step():
#         with tf.GradientTape(watch_accessed_variables=False) as tape:
#             tape.watch(model.trainable_variables)
#             objective = model.elbo(*data)
#             grads = tape.gradient(objective, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         return -objective
#
#     for step in range(iterations):
#         elbo = optimization_step()
#         print(f"{step}\t{elbo:.4f}", end="\r")
#         if callback is not None:
#             callback(step)
#     print("")
#
#     return logf


def normalize(X, X_mean, X_std):
    return (X - X_mean) / X_std


@dataclass
class Experiment:
    storage_path: str
    base_filename: Optional[str] = "data"

    # Populated during object life
    model = None
    trained_parameters = None

    _X_train = None
    _Y_train = None
    _X_test = None
    _Y_test = None

    def load_data(self):
        raise NotImplementedError

    def setup_model(self):
        """
        Set up the model here to the point where existing parameters can be loaded into it. Do not
        initialise the parameters, as this can be time consuming.
        :return:
        """
        raise NotImplementedError

    def init_params(self):
        """
        Do the time consuming parameter initialisation here.
        :return:
        """
        raise NotImplementedError

    def run_optimisation(self):
        raise NotImplementedError

    def run(self):
        self.setup_model()
        self.init_params()
        self.run_optimisation()

    def cached_run(self):
        try:
            self.load()
            print("Skipping...")
        except FileNotFoundError:
            self.run()
            self.save()

    @property
    def X_train(self):
        if self._X_train is None:
            self.load_data()
        return self._X_train

    @property
    def Y_train(self):
        if self._Y_train is None:
            self.load_data()
        return self._Y_train

    @property
    def X_test(self):
        if self._X_test is None:
            self.load_data()
        return self._X_test

    @property
    def Y_test(self):
        if self._Y_test is None:
            self.load_data()
        return self._Y_test

    @property
    def store_variables(self):
        return [
            k for k in list(self.__dict__.keys()) if k[0] != "_" and k not in ["storage_path", "base_filename", "model"]
        ]

    @property
    def load_match_variables(self):
        return [k for k in self.store_variables if k not in ["trained_parameters"]]

    def save(self):
        store_dict = {k: v for k, v in self.__dict__.items() if k in self.store_variables}
        filename = get_next_filename(self.storage_path, self.base_filename, extension="json")
        json_tricks.dump(store_dict, filename)
        print(f"Stored results in {filename} at {datetime.datetime.now()}")

    def load(self, filename=None):
        def field_equal(a, b):
            if type(a) is dict:
                if a == {} and type(b) is _MISSING_TYPE:
                    return True
                try:
                    equality = True
                    for k in reduce(set.union, map(set, map(dict.keys, [a, b]))):
                        if type(a[k]) is np.ndarray:
                            if not np.all(a[k] == b[k]):
                                return False
                        else:
                            if a[k] != b[k]:
                                return False
                except (TypeError, KeyError):
                    equality = False

                return equality
            else:
                return a == b

        if filename is None:
            # Find run with similar run parameters
            existing_runs = []
            for fn in glob(f"{self.storage_path}/{self.base_filename}*"):
                existing_runs.append((json_tricks.load(fn), fn))

            matching_runs = [
                (dict, fn)
                for dict, fn in existing_runs
                if all(
                    [
                        field_equal(self.__dict__[k], (dict[k] if k in dict else self.__dataclass_fields__[k].default))
                        for k in self.load_match_variables
                    ]
                )
            ]
        else:
            matching_runs = [(json_tricks.load(filename), filename)]

        if len(matching_runs) == 1:
            print(f"Loading from `{matching_runs[0][1]}`...")
            for k, v in matching_runs[0][0].items():
                setattr(self, k, v)
            gpflow.config.set_default_positive_minimum(1e-7)
            self.setup_model()
            gpflow.utilities.multiple_assign(self.model, self.trained_parameters)
        elif len(matching_runs) == 0:
            raise FileNotFoundError("No matching run found.")
        else:
            raise AssertionError("Only one run of an experiment should be present.")


@dataclass
class UciExperiment(Experiment):
    dataset_name: Optional[str] = "Wilson_elevators"

    def load_data(self):
        loaded_data = getattr(data, self.dataset_name)()
        if type(loaded_data) == tuple:
            self._X_train, self._Y_train = loaded_data[0]
        elif isinstance(loaded_data, data.Dataset):
            # Here, we always normalise on training. This is different to before.
            X_mean, X_std = np.average(loaded_data.X_train, 0)[None, :], 1e-6 + np.std(loaded_data.X_train, 0)[None, :]
            self._X_train = normalize(loaded_data.X_train, X_mean, X_std)
            self._X_test = normalize(loaded_data.X_test, X_mean, X_std)

            Y_mean, Y_std = np.average(loaded_data.Y_train, 0)[None, :], 1e-6 + np.std(loaded_data.Y_train, 0)[None, :]
            self._Y_train = normalize(loaded_data.Y_train, Y_mean, Y_std)
            self._Y_test = normalize(loaded_data.Y_test, Y_mean, Y_std)
        else:
            raise NotImplementedError


@dataclass
class GaussianProcessUciExperiment(UciExperiment):
    model_class: Optional[str] = "SGPR"
    M: Optional[int] = None
    kernel_name: Optional[str] = "SquaredExponential"
    init_Z_method: Optional[InducingPointInitializer] = FirstSubsample()
    max_lengthscale: Optional[float] = 1000.0
    max_variance: Optional[float] = 1000.0

    training_procedure: Optional[str] = "joint"  # joint | reinit
    initial_parameters: Optional[dict] = field(default_factory=dict)

    # Populated during object life
    train_objective_hist = None

    def setup_model(self):
        kernel = self.setup_kernel()
        if self.model_class == "SGPR":
            inducing_variable = self.setup_inducing_variable()
            model = RobustSGPR((self.X_train, self.Y_train), kernel, inducing_variable=inducing_variable)
        elif self.model_class == "GPR":
            assert self.M is None
            model = RobustGPR((self.X_train, self.Y_train), kernel)
        else:
            raise NotImplementedError
        print(f"Jitter variance: {np.log10(model.jitter_variance.numpy()):.1f}")
        model.likelihood.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.model = model

    def setup_kernel(self):
        if self.kernel_name == "SquaredExponential":
            kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(self.X_train.shape[1]))
        elif self.kernel_name == "SquaredExponentialLinear":
            kernel = (
                    gpflow.kernels.SquaredExponential(
                        lengthscales=np.ones(self.X_train.shape[1])) + gpflow.kernels.Linear()
            )
        else:
            # try:
            kernel = getattr(gpflow.kernels, self.kernel_name)(lengthscales=np.ones(self.X_train.shape[1]))
            # except:
            #     raise NotImplementedError(f"Kernel `{self.kernel_name}` is unknown.")

        return kernel

    def setup_inducing_variable(self):
        return gpflow.inducing_variables.InducingPoints(np.zeros((self.M, self.X_train.shape[1])))

    def init_inducing_variable(self):
        if self.M > len(self.X_train):
            raise ValueError("Cannot have M > len(X).")

        Z, _ = self.init_Z_method(self.X_train, self.M, self.model.kernel)

        try:
            self.model.inducing_variable.Z.assign(Z)
        except Exception as e:
            print(type(e))
            print(e)
            self.model.inducing_variable.Z = gpflow.Parameter(Z)

    def init_params(self):
        self.model.likelihood.variance.assign(0.01)
        gpflow.utilities.multiple_assign(self.model, self.initial_parameters)

        constrained_transform = tfp.bijectors.Sigmoid(
            gpflow.utilities.to_default_float(gpflow.config.default_positive_minimum()),
            gpflow.utilities.to_default_float(self.max_lengthscale),
        )

        var_constrained_transform = tfp.bijectors.Sigmoid(
            gpflow.utilities.to_default_float(gpflow.config.default_positive_minimum()),
            gpflow.utilities.to_default_float(self.max_variance),
        )

        if self.kernel_name == "SquaredExponential":
            new_len = gpflow.Parameter(self.model.kernel.lengthscales.numpy(), transform=constrained_transform)
            new_var = gpflow.Parameter(self.model.kernel.variance.numpy(), transform = var_constrained_transform)
            self.model.kernel.lengthscales = new_len
            self.model.kernel.variance = new_var
        elif self.kernel_name == "SquaredExponentialLinear":
            new_len = gpflow.Parameter(self.model.kernel.kernels[0].lengthscales.numpy(),
                                       transform=constrained_transform)
            self.model.kernel.kernels[0].lengthscales = new_len
            new_var_se = gpflow.Parameter(self.model.kernel[0].variance.numpy(), transform=var_constrained_transform)
            new_var_lin = gpflow.Parameter(self.model.kernel[1].variance.numpy(), transform=var_constrained_transform)
            self.model.kernel[0].variance = new_var_se
            self.model.kernel[1].variance = new_var_lin

        # TODO: Check if "inducing_variable" is in one of the keys in `self.initial_parameters`, to make things work
        #       with non `InducingPoints` like inducing variables.
        if self.model_class != "GPR" and ".inducing_variable.Z" not in self.initial_parameters:
            # Kernel parameters should be initialised before inducing variables are. If inducing variables are set in
            # the initial parameters, we shouldn't run this.
            self.init_inducing_variable()


@dataclass
class FullbatchUciExperiment(GaussianProcessUciExperiment):
    optimizer: Optional[str] = "l-bfgs-b"
    training_procedure: Optional[str] = "joint"  # joint | reinit

    def run_optimisation(self):
        print(f"Running {str(self)}")

        model = self.model
        loss_function = self.model.training_loss_closure(compile=True)
        robust_loss_function = lambda: -self.model.robust_maximum_log_likelihood_objective()
        # loss_function = tf.function(lambda jitter=None: -self.model.elbo(jitter))
        hist = LoggerCallback(model, robust_loss_function)
        if self.optimizer == "l-bfgs-b" or self.optimizer == "bfgs":
            opt = RobustScipy()
        else:
            raise NotImplementedError(f"I don't know {self.optimizer}")
        def run_optimisation():
            try:
                opt.minimize(
                    loss_function,
                    self.model.trainable_variables,
                    robust_closure=robust_loss_function,
                    method=self.optimizer,
                    options=dict(maxiter=1000, disp=True),
                    step_callback=hist,
                )
                print("")
            except KeyboardInterrupt as e:
                if input("Optimisation aborted. Do you want to re-raise the KeyboardInterrupt? (y/n) ") == "y":
                    raise e

        if self.training_procedure == "joint":
            run_optimisation()
        elif self.training_procedure == "fixed_Z":
            set_trainable(self.model.inducing_variable, False)
            run_optimisation()
            run_optimisation()
        elif self.training_procedure == "reinit_Z":
            set_trainable(self.model.inducing_variable, False)
            for i in range(20):
                reinit = True
                try:
                    run_optimisation()
                except tf.errors.InvalidArgumentError as e:
                    if e.message[1:9] != "Cholesky":
                        raise e
                    self.init_inducing_variable()
                    print(self.model.elbo().numpy())  # Check whether Cholesky fails
                    reinit = False

                if reinit:
                    old_Z = self.model.inducing_variable.Z.numpy().copy()
                    old_elbo = self.model.robust_maximum_log_likelihood_objective()
                    self.init_inducing_variable()
                    if self.model.robust_maximum_log_likelihood_objective() <= old_elbo:
                        # Restore old Z, and finish optimisation
                        self.model.inducing_variable.Z.assign(old_Z)
                        print("Stopped reinit_Z procedure because new ELBO was smaller than old ELBO.")
                        break
        else:
            raise NotImplementedError

        # Store results
        self.trained_parameters = gpflow.utilities.read_values(model)
        self.train_objective_hist = opt.f_vals  #(hist.n_iters, hist.log_likelihoods)
