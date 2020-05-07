# # Inducing point initialisation while optimising hypers
# Assess how well inducing point initialisation works, in conjunction with optimising the hyperparameters.
# Here, we compare:
#  - Fixed Z,       initialised with the baseline kernel hyperparameters
#  - EM "reinit" Z, initialised with the baseline kernel hyperparameters
# In all cases, the Z are initialised using default kernel parameters, not the ones from the baseline. This is to ensure
# that we do not use information when initialising Z that isn't accessable when running a new dataset.
#
# Local optima are a bit annoying when "cold" initialising. They make the plot appear non-smooth. So we initialise at


from dataclasses import dataclass
from typing import Optional

import jug
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp

import gpflow
import inducing_init
from inducing_init_experiments.init_z.utils import uci_train_settings, print_post_run
from inducing_init_experiments.utils import baselines, FullbatchUciExperiment

#
#
# Settings
# dataset_name = "Naval_noisy"
dataset_name = "Wilson_concrete"
# init_from_baseline = True
init_from_baseline = False

#
#
# Setup
gpflow.config.set_default_positive_minimum(1.0e-5)
gpflow.config.set_default_jitter(1e-10)

init_Z_methods = [
    inducing_init.FirstSubsample(),
    inducing_init.ConditionalVariance(),
    inducing_init.ConditionalVariance(sample=True),
    inducing_init.Kmeans(),
]
experiment_name = "init-inducing"

init_Z_before_hypers = init_from_baseline  # Only if you init from baseline do you need to init Z before hypers

# %%
experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"

common_run_settings = dict(storage_path=experiment_storage_path, dataset_name=dataset_name, max_lengthscale=1001.0,
                           training_procedure="fixed_Z")

uci_train_settings.update(dict(
    Naval_noisy=([10, 20, 30, 40, 45, 47, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100,
                  130, 150, 180, 200, 250, 300, 400, 500], {}),  # Very sparse solution exists
    # Naval_noisy=([10, 20, 50, 100, 200, 500], {}),  # Very sparse solution exists
))
Ms, dataset_custom_settings = uci_train_settings[dataset_name]

dataset_plot_settings = dict(
    Naval_noisy=dict(elbo_ylim=(-20e3, 45e3))
).get(dataset_name, dict(elbo_ylim=None))

baseline_custom_settings = dict(
    Naval_noisy={"model_class": "SGPR", "M": 1000, "training_procedure": "reinit_Z",
                 "init_Z_method": inducing_init.ConditionalVariance(sample=False), "max_lengthscale": 1000.0}
).get(dataset_name, dict(model_class="GPR", training_procedure="joint", max_lengthscale=1000.0))


@dataclass
class InitZBeforeHypers(FullbatchUciExperiment):
    init_Z_before_params: Optional[str] = False

    def init_params(self):
        self.model.likelihood.variance.assign(0.01)
        if not self.init_Z_before_params:
            gpflow.utilities.multiple_assign(self.model, self.initial_parameters)

        constrained_transform = tfp.bijectors.Sigmoid(
            gpflow.utilities.to_default_float(gpflow.config.default_positive_minimum()),
            gpflow.utilities.to_default_float(self.max_lengthscale),
        )

        if self.kernel_name == "SquaredExponential":
            new_len = gpflow.Parameter(self.model.kernel.lengthscales.numpy(), transform=constrained_transform)
            self.model.kernel.lengthscales = new_len
        elif self.kernel_name == "SquaredExponentialLinear":
            new_len = gpflow.Parameter(self.model.kernel[0].lengthscales.numpy(), transform=constrained_transform)
            self.model.kernel.kernels[0].lengthscales = new_len

        # TODO: Check if "inducing_variable" is in one of the keys in `self.initial_parameters`, to make things work
        #       with non `InducingPoints` like inducing variables.
        if self.model_class != "GPR" and ".inducing_variable.Z" not in self.initial_parameters:
            # Kernel parameters should be initialised before inducing variables are. If inducing variables are set in
            # the initial parameters, we shouldn't run this.
            self.init_inducing_variable()

        if self.init_Z_before_params:
            gpflow.utilities.multiple_assign(self.model, self.initial_parameters)

def compute_model_stats(exp):
    rmse = np.mean((exp.model.predict_f(exp.X_test)[0].numpy() - exp.Y_test) ** 2.0) ** 0.5
    nlpp = -np.mean(exp.model.predict_log_density((exp.X_test, exp.Y_test)))
    elbo = exp.model.elbo().numpy()
    upper = exp.model.upper_bound().numpy()
    return elbo, upper, rmse, nlpp

# Baseline runs
print("Baseline exp...")
baseline_exp = FullbatchUciExperiment(**{**common_run_settings, **dataset_custom_settings, **baseline_custom_settings})

@jug.TaskGenerator
def run_baseline(baseline_exp):
    baseline_exp.cached_run()
    if baseline_exp.model_class == "SGPR":
        baseline_lml = baseline_exp.model.elbo().numpy()
    else:
        baseline_lml = baseline_exp.model.log_marginal_likelihood().numpy()
    model_parameters = gpflow.utilities.read_values(baseline_exp.model) if init_from_baseline else {}
    if ".inducing_variable.Z" in model_parameters:
        model_parameters.pop(".inducing_variable.Z")
    full_rmse = np.mean(
        (baseline_exp.model.predict_f(baseline_exp.X_test)[0].numpy() - baseline_exp.Y_test) ** 2.0) ** 0.5
    full_nlpp = -np.mean(baseline_exp.model.predict_log_density((baseline_exp.X_test, baseline_exp.Y_test)))

    return model_parameters, full_rmse, full_nlpp, baseline_lml

model_parameters, full_rmse, full_nlpp, baseline_lml = jug.bvalue(run_baseline(baseline_exp))

# Bound optimisation
@jug.TaskGenerator
def run_sparse_opt(exp):
    print(exp)
    exp.cached_run()
    print_post_run(exp)
    elbo, upper, rmse, nlpp = compute_model_stats(exp)
    return elbo, upper, rmse, nlpp


# Sparse runs
init_Z_runs = {}
init_Z_task_results = {}
for init_Z_method in init_Z_methods:
    settings_for_runs = [{"model_class": "SGPR", "M": M, "init_Z_method": init_Z_method,
                          "base_filename": "opthyp-fixed_Z", "initial_parameters": model_parameters,
                          "init_Z_before_params": init_Z_before_hypers, **dataset_custom_settings}
                         for M in Ms]
    init_Z_runs[str(init_Z_method)] = []
    init_Z_task_results[str(init_Z_method)] = []
    for run_settings in settings_for_runs:
        exp = InitZBeforeHypers(**{**common_run_settings, **run_settings})
        result = run_sparse_opt(exp)
        init_Z_runs[str(init_Z_method)].append(exp)
        init_Z_task_results[str(init_Z_method)].append(result)


#
#
# Optimisation of Z
settings_for_runs = [{"model_class": "SGPR", "M": M,
                      "training_procedure": "reinit_Z", "base_filename": "opthyp-reinit_Z",
                      "initial_parameters": model_parameters,
                      "init_Z_before_params": init_Z_before_hypers, **dataset_custom_settings}
                     for M in Ms]
for name, init_Z_method in zip(
        ["reinit_Z_sF", "reinit_Z_sT"],
        [inducing_init.ConditionalVariance(sample=False), inducing_init.ConditionalVariance(sample=True)]
):
    init_Z_runs[name] = []
    init_Z_task_results[name] = []
    for run_settings in settings_for_runs:
        exp = InitZBeforeHypers(**{**common_run_settings, **run_settings, "init_Z_method": init_Z_method})
        result = run_sparse_opt(exp)
        init_Z_runs[name].append(exp)
        init_Z_task_results[name].append(result)

