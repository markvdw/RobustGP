from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from . import data
from .storing import store_pickle, load_existing_runs


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
            self.log_likelihoods.append(self.loss_function().numpy())
            print(f"{self.counter} - objective function: {self.log_likelihoods[-1]:.4f}", end="\r")

        self.counter += 1


def run_tf_optimizer(model, optimizer, data, iterations, callback=None):
    logf = []

    @tf.function(autograph=False)
    def optimization_step():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            objective = model.elbo(*data)
            grads = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return -objective

    for step in range(iterations):
        elbo = optimization_step()
        print(f"{step}\t{elbo:.4f}", end="\r")
        if callback is not None:
            callback(step)
    print("")

    return logf


def greedy_trace_init(kernel, X, M):
    """
    USING Lightly modified CODE FROM:
    https://papers.nips.cc/paper/7805-fast-greedy-map-inference-for-
      determinantal-point-process-to-improve-recommendation-diversity.pdf
    Github source:
    https://github.com/laming-chen/fast-map-dpp
    :param kernel: gpflow.kernel
    :param M: maximum number of inducing points
    :return: list
    """
    print("greedy init")
    cis = np.zeros((M, X.shape[0]))
    di2s = kernel.K_diag(X)
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < M:
        m = len(selected_items) - 1
        ci_optimal = cis[:m, selected_item]
        di_optimal = np.sqrt(di2s[selected_item])
        new_X = X[selected_item:selected_item + 1, :]
        elements = kernel.K(new_X, X)[0, :]
        eis = (elements - np.dot(ci_optimal, cis[:m, :])) / di_optimal
        cis[m, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(selected_item)
        if np.sum(di2s) < 1e-10:  # I added this, break optimization if t<1e-10
            break
    return X[selected_items, :]


def normalize(X, X_mean, X_std):
    return (X - X_mean) / X_std


class Experiment:
    def run(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class ExperimentRecord(Experiment):
    # = Experiment settings
    storage_path: str
    dataset_name: str

    # - Training set parameters
    normalise_on_training: Optional[bool] = False

    # - Model setup parameters
    model_class: Optional[str] = "SVGP"
    model_kwargs: Optional[dict] = None
    kernel_class: Optional[str] = "SquaredExponential"
    lengthscale_transform: Optional[str] = "positive"

    # - Approximation parameters
    M: Optional[int] = 100
    init_Z_method: Optional[str] = "first"

    # - Optimisation parameters
    cg_pre_optimization: Optional[bool] = False
    optimizer: Optional[str] = "l-bfgs-b"
    learning_rate: Optional[float] = None
    fixed_Z: Optional[bool] = False
    restart_id: Optional[int] = 0

    # Populated after running the experiment
    _X_train = None
    _Y_train = None
    _X_test = None
    _Y_test = None
    model = None
    train_objective_hist = None
    trained_parameters = None

    def _load_data(self):
        loaded_data = getattr(data, self.dataset_name)()
        if type(loaded_data) == tuple:
            self._X_train, self._Y_train = loaded_data[0]
        elif isinstance(loaded_data, data.Dataset):
            if self.normalise_on_training:
                X_mean, X_std = np.average(loaded_data.X_train, 0)[None, :], 1e-6 + np.std(loaded_data.X_train, 0)[None,
                                                                                    :]
                self._X_train = normalize(loaded_data.X_train, X_mean, X_std)
                self._X_test = normalize(loaded_data.X_test, X_mean, X_std)

                Y_mean, Y_std = np.average(loaded_data.Y_train, 0)[None, :], 1e-6 + np.std(loaded_data.Y_train, 0)[None,
                                                                                    :]
                self._Y_train = normalize(loaded_data.Y_train, Y_mean, Y_std)
                self._Y_test = normalize(loaded_data.Y_test, Y_mean, Y_std)
            else:
                self._X_train, self._Y_train = loaded_data.X_train, loaded_data.Y_train
                self._X_test, self._Y_test = loaded_data.X_test, loaded_data.Y_test
        else:
            raise NotImplementedError

    @property
    def X_train(self):
        if self._X_train is None:
            self._load_data()
        return self._X_train

    @property
    def Y_train(self):
        if self._Y_train is None:
            self._load_data()
        return self._Y_train

    @property
    def X_test(self):
        if self._X_test is None:
            self._load_data()
        return self._X_test

    @property
    def Y_test(self):
        if self._Y_test is None:
            self._load_data()
        return self._Y_test

    def cached_run(self, maxiter, init_from_model=None):
        try:
            self.load()
            print("Skipping...")
        except FileNotFoundError:
            self.run(maxiter, init_from_model=init_from_model)
            self.save()

    def _get_modelclass_from_name(self):
        return {"SGPR": gpflow.models.SGPR, "SVGP": gpflow.models.SVGP, "GPR": gpflow.models.GPR}[self.model_class]

    def _init_Z(self, X, kernel=None):
        if self.M > len(X):
            raise ValueError("Cannot have M > len(X).")
        if self.init_Z_method == "first":
            return X[:self.M, :].copy()
        elif self.init_Z_method == "uniform":
            return X[np.random.permutation(len(X))[:self.M], :].copy()
        elif self.init_Z_method == "greedy-trace":
            return greedy_trace_init(kernel, X, self.M)
        else:
            raise NotImplementedError

    def _init_model(self, init_from_model=None, load=False):
        if not load:
            old_positive_minimum = gpflow.config.default_positive_minimum()
            gpflow.config.set_default_positive_minimum(old_positive_minimum * 1.01)
        modelclass = self._get_modelclass_from_name()
        if self.kernel_class == "SquaredExponential":
            kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(self.X_train.shape[1]))
            if self.lengthscale_transform == "positive":
                pass
            elif self.lengthscale_transform == "constrained":
                factor = 0.0 if load else 0.01
                constrained_transform = tfp.bijectors.Sigmoid(
                    gpflow.utilities.to_default_float(gpflow.config.default_positive_minimum() * (1.0 + factor)),
                    gpflow.utilities.to_default_float(10.0 * (1.0 - factor))
                )
                new_len = gpflow.Parameter(kernel.lengthscales.numpy(), name=kernel.lengthscales.name.split(':')[0],
                                           transform=constrained_transform)
                kernel.lengthscales = new_len
            else:
                raise NotImplementedError
        elif self.kernel_class == "Linear":
            kernel = gpflow.kernels.Linear(np.ones(self.X_train.shape[1]))
        else:
            raise NotImplementedError
        init_Z = self._init_Z(self.X_train, kernel)
        if modelclass == gpflow.models.SGPR:
            model = modelclass((self.X_train, self.Y_train), kernel, inducing_variable=init_Z)
        elif modelclass == gpflow.models.GPR:
            model = modelclass((self.X_train, self.Y_train), kernel)
        else:
            model = modelclass(kernel, gpflow.likelihoods.Gaussian(), init_Z, **self.model_kwargs)

        model.likelihood.variance.assign(0.01)
        model.kernel.lengthscales.assign(self.X_train.shape[1] ** 0.5 * np.ones(self.X_train.shape[1]))

        if init_from_model is not None:
            # Initialise from parameters from init_from_model
            print("Initialising from previous model...")
            init_dict = gpflow.utilities.read_values(init_from_model)
            init_dict.pop(".inducing_variable.Z")
            init_dict['.kernel.lengthscales'] = np.clip(init_dict['.kernel.lengthscales'], 1e-5, 999.0)
            gpflow.utilities.multiple_assign(model, init_dict)

        self.model = model

        if not load:
            gpflow.config.set_default_positive_minimum(old_positive_minimum)

    def _create_loss_function(self, training_dataset):
        if self.model_class in ["SGPR", "GPR"]:
            # @tf.function(autograph=False)
            def loss():
                return -self.model.log_likelihood()
        else:
            @tf.function(autograph=False)
            def loss():
                return -self.model.log_likelihood(*training_dataset)

        return loss

    def run(self, maxiter, init_from_model=None):
        print(f"Running {str(self)}")

        self._init_model(init_from_model)
        model = self.model

        if hasattr(model, 'inducing_variable'):
            gpflow.utilities.set_trainable(model.inducing_variable.Z, not self.fixed_Z)

        loss_function = self.model.training_loss_closure(compile=True)
        hist = LoggerCallback(model, loss_function)
        if self.optimizer == "l-bfgs-b" or self.optimizer == "bfgs":
            try:
                print("Pre optimisation")
                if self.cg_pre_optimization:
                    opt = gpflow.optimizers.Scipy()
                    opt.minimize(loss_function, self.model.trainable_variables, method="CG",
                                 options=dict(maxiter=30, disp=False), step_callback=hist)

                print("")
                print("Full optimisation")
                opt = gpflow.optimizers.Scipy()
                opt.minimize(loss_function, self.model.trainable_variables, method=self.optimizer,
                             options=dict(maxiter=10000, disp=False), step_callback=hist)
                print("")
            except KeyboardInterrupt:
                pass  # TOOD: Come up with something better than just pass...
        elif self.optimizer == "adam":
            # adam = tf.optimizers.Adam(learning_rate=run_details["learning_rate"])
            # run_tf_optimizer(model, adam, (X, Y), maxiter, hist)
            pass
        else:
            raise NotImplementedError(f"I don't know {self.optimizer}")

        # Store results
        self.trained_parameters = gpflow.utilities.read_values(model)
        self.train_objective_hist = (hist.n_iters, hist.log_likelihoods)

    def save(self):
        store_pickle({k: v for k, v in self.__dict__.items() if k not in ["model", "training_loss"]}, self.storage_path)

    def load(self):
        existing_runs = load_existing_runs(self.storage_path)
        # Fields in this object need to match the ones that are stored. If a field exists here that is not stored, it
        # must have the default value.
        compare_keys = [k for k in self.__annotations__.keys() if k not in ['storage_path', 'dataset_name']]
        matching_runs = [(d, fp) for d, fp in existing_runs
                         if all([self.__dict__[k] == (d[k] if k in d else self.__dataclass_fields__[k].default)
                                 for k in compare_keys])]
        if len(matching_runs) == 1:
            print(f"Loading from `{matching_runs[0][1]}`...")
            for k, v in matching_runs[0][0].items():
                setattr(self, k, v)
            self._init_model(load=True)
            gpflow.utilities.multiple_assign(self.model, self.trained_parameters)
        elif len(matching_runs) == 0:
            raise FileNotFoundError("No matching run found.")
        else:
            raise AssertionError("Only one run of an experiment should be present.")
