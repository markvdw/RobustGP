# # Inducing point initlaisation while optimising hypers
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
dataset_name = "Wilson_gas"
# init_from_baseline = True
init_from_baseline = False

#
#
# Setup
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

#
#
# Baseline runs
print("Baseline run...")
baseline_exp = FullbatchUciExperiment(**{**common_run_settings, **dataset_custom_settings, **baseline_custom_settings})
baseline_exp.cached_run()
if baseline_exp.model_class == "SGPR":
    baseline_lml = baseline_exp.model.elbo().numpy()
else:
    baseline_lml = baseline_exp.model.log_marginal_likelihood().numpy()
model_parameters = gpflow.utilities.read_values(baseline_exp.model) if init_from_baseline else {}
if ".inducing_variable.Z" in model_parameters:
    model_parameters.pop(".inducing_variable.Z")


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


#
#
# Sparse runs
init_Z_runs = {}
for init_Z_method in init_Z_methods:
    settings_for_runs = [{"model_class": "SGPR", "M": M, "init_Z_method": init_Z_method,
                          "base_filename": "opthyp-fixed_Z", "initial_parameters": model_parameters,
                          "init_Z_before_params": init_Z_before_hypers, **dataset_custom_settings}
                         for M in Ms]
    init_Z_runs[str(init_Z_method)] = []
    for run_settings in settings_for_runs:
        run = InitZBeforeHypers(**{**common_run_settings, **run_settings})
        run.cached_run()
        print_post_run(run)
        init_Z_runs[str(init_Z_method)].append(run)

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
    for run_settings in settings_for_runs:
        run = InitZBeforeHypers(**{**common_run_settings, **run_settings, "init_Z_method": init_Z_method})
        run.cached_run()
        print_post_run(run)

        init_Z_runs[name].append(run)

#
#
# Evaluation
full_rmse = np.mean((baseline_exp.model.predict_f(baseline_exp.X_test)[0].numpy() - baseline_exp.Y_test) ** 2.0) ** 0.5
full_nlpp = -np.mean(baseline_exp.model.predict_log_density((baseline_exp.X_test, baseline_exp.Y_test)))

init_Z_rmses = {}
init_Z_nlpps = {}
init_Z_elbos = {}
init_Z_uppers = {}
for init_Z_method in init_Z_runs.keys():
    init_Z_rmses[init_Z_method] = []
    init_Z_nlpps[init_Z_method] = []
    init_Z_elbos[init_Z_method] = []
    init_Z_uppers[init_Z_method] = []
    for run in init_Z_runs[init_Z_method]:
        rmse = np.mean((run.model.predict_f(run.X_test)[0].numpy() - run.Y_test) ** 2.0) ** 0.5
        init_Z_rmses[init_Z_method].append(rmse)
        nlpp = -np.mean(run.model.predict_log_density((run.X_test, run.Y_test)))
        init_Z_nlpps[init_Z_method].append(nlpp)
        init_Z_elbos[init_Z_method].append(run.model.elbo().numpy())
        init_Z_uppers[init_Z_method].append(run.model.upper_bound().numpy())

m_elbo, m_rmse, m_nlpp = baselines.meanpred_baseline(None, baseline_exp.Y_train, None, baseline_exp.Y_test)
l_elbo, l_rmse, l_nlpp = baselines.linear_baseline(baseline_exp.X_train, baseline_exp.Y_train,
                                                   baseline_exp.X_test, baseline_exp.Y_test)

#
#
# Plotting
print(f"gpr rmse: {full_rmse}")
print(f"rmse    : {init_Z_rmses}")
print(f"gpr nlpp: {full_nlpp}")
print(f"nlpp    : {init_Z_nlpps}")

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    l, = ax.plot(Ms, init_Z_elbos[method], label=f"{method} elbo")
    ax.plot(Ms, init_Z_uppers[method], label=f"{method} upper", color=l.get_color(), linestyle=(0, (3, 1, 1, 1, 1, 1)))
ax.axhline(baseline_lml, label='full GP', linestyle="--")
ax.axhline(l_elbo, label='linear', linestyle='-.')
ax.axhline(m_elbo, label='mean', linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("elbo")
ax.set_ylim(dataset_plot_settings["elbo_ylim"])
fig.savefig(f"./figures/opthyp-{dataset_name}-{init_from_baseline}-elbo.png")

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.plot(Ms, init_Z_rmses[method], label=method)
ax.axhline(full_rmse, label="full GP", linestyle='--')
ax.axhline(l_rmse, label="linear", linestyle='-.')
ax.axhline(m_rmse, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("rmse")
fig.savefig(f"./figures/opthyp-{dataset_name}-{init_from_baseline}-rmse.png")

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.plot(Ms, init_Z_nlpps[method], label=method)
ax.axhline(full_nlpp, label="full GP", linestyle='--')
ax.axhline(l_nlpp, label="linear", linestyle='-.')
ax.axhline(m_nlpp, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("nlpp")
fig.savefig(f"./figures/opthyp-{dataset_name}-{init_from_baseline}-nlpp.png")

plt.show()
