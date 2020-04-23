# # Inducing point initlaisation fixed hyperparameters
# Assess how well inducing point initialisation works, with the hyperparameters fixed to the ones found by the full GP.
# This simplifies things, since we only need to run optimisation with the full GP (or a GP with many inducing points)

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from inducing_experiments.utils import baselines, FullbatchUciExperiment, LoggerCallback

gpflow.config.set_default_positive_minimum(1.0e-5)
gpflow.config.set_default_jitter(1e-10)

init_Z_methods = ["first", "greedy-trace"]
experiment_name = "init-inducing-fixedhyp"
# dataset_name = "Naval_noisy"
dataset_name = "Wilson_gas"

# %%
experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"

common_run_settings = dict(storage_path=experiment_storage_path, dataset_name=dataset_name, max_lengthscale=1001.0)

Ms, dataset_custom_settings = dict(
    Naval_noisy=([10, 20, 30, 40, 45, 47, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100,
                  130, 150, 180, 200, 250, 300, 400, 500], {}),  # Very sparse solution exists
    Wilson_gas=([100, 200, 500, 1000, 1300], {}),
    # Naval_noisy=([10, 20, 50, 100, 200, 500], {}),  # Very sparse solution exists

    Wilson_pol=([100, 200, 500, 1000, 2000], {}),
    Naval=([10, 20, 50, 100, 200], {}),  # Very sparse solution exists
    Power=([100, 200, 500, 1000, 2000], {}),  # Step function in it?
    Kin8mn=([100, 200, 500, 1000, 2000], {}),  # Can't download
    Wilson_parkinsons=([100, 200, 500, 1000], {"max_lengthscale": 10.0}),  # Cholesky errors
    Wilson_sml=([100, 200, 500, 1000, 2000, 3000, 3500], {}),  # Mostly linear, but with benefit of nonlinear
    # Didn't get SE+Lin working, probably local optimum
    # Wilson_skillcraft=([10, 20, 50, 100, 200, 500], {"kernel_name": "SquaredExponentialLinear"}),
    Wilson_skillcraft=([10, 20, 50, 100, 200, 500, 1000], {}),  # Mostly linear, but with benefit of nonlinear
    Wilson_wine=([100, 200, 500, 1000, 1300], {}),  # Suddenly catches good hypers with large M
    Wilson_airfoil=([100, 200, 500, 1000, 1250, 1300, 1340], {}),  # Good
    Wilson_solar=([100, 200, 300],
                  {"kernel_name": "SquaredExponentialLinear", "max_lengthscale": 10.0}),  # Mostly linear
    # Good, better performance with Linear kernel added
    # Wilson_concrete=([100, 200, 500, 600, 700, 800, 900],
    #                  {"kernel_name": "SquaredExponentialLinear", "optimizer": "bfgs", "max_lengthscale": 10.0}),
    Wilson_concrete=([100, 200, 500, 600, 700, 800, 900], {}),
    Wilson_pendulum=([10, 100, 200, 500, 567], {}),  # Reasonable, very low noise
    Wilson_forest=([10, 100, 200, 400], {"kernel_name": "SquaredExponentialLinear"}),  # Bad
    Wilson_energy=([10, 50, 100, 200, 500], {}),  # Good
    Wilson_stock=([10, 50, 100, 200, 400, 450], {"kernel_name": "SquaredExponentialLinear"}),  # Mostly linear
    Wilson_housing=([100, 200, 300, 400], {})  # Bad
)[dataset_name]

dataset_plot_settings = dict(
    Naval_noisy=dict(elbo_ylim=(-20e3, 45e3))
).get(dataset_name, dict(elbo_ylim=None))

baseline_custom_settings = dict(
    Naval_noisy={"model_class": "SGPR", "M": 1000, "training_procedure": "reinit_Z",
                 "init_Z_method": "greedy-trace", "max_lengthscale": 1000.0}
).get(dataset_name, dict(model_class="GPR", max_lengthscale=1000.0))


def print_post_run(run):
    print("")
    try:
        std_ratio = (run.model.kernel.variance.numpy() / run.model.likelihood.variance.numpy()) ** 0.5
        print(f"(kernel.variance / likelihood.variance)**0.5: {std_ratio}")
        print(run.model.kernel.lengthscales.numpy())
        print(f"ELBO: {run.model.elbo().numpy()}")
    except AttributeError:
        pass
    print("")
    print("")


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
model_parameters = gpflow.utilities.read_values(baseline_exp.model)
if ".inducing_variable.Z" in model_parameters:
    model_parameters.pop(".inducing_variable.Z")


@dataclass
class FullbatchUciInducingOptExperiment(FullbatchUciExperiment):
    optimise_objective: Optional[str] = None  # None | lower | upper
    initial_parameters: Optional[dict] = field(default_factory=dict)

    def init_model(self):
        gpflow.utilities.multiple_assign(self.model, self.initial_parameters)

    def run_optimisation(self):
        print(f"Running {str(self)}")

        model = self.model
        gpflow.utilities.set_trainable(model, False)
        gpflow.utilities.set_trainable(model.inducing_variable, True)

        if self.optimise_objective is None:
            return
        elif self.optimise_objective == "upper":
            loss_function = tf.function(model.upper_bound)
        elif self.optimise_objective == "lower":
            loss_function = self.model.training_loss_closure(compile=True)
        else:
            raise NotImplementedError
        hist = LoggerCallback(model, loss_function)

        def run_optimisation():
            if self.optimizer == "l-bfgs-b" or self.optimizer == "bfgs":
                try:
                    opt = gpflow.optimizers.Scipy()
                    opt.minimize(loss_function, self.model.trainable_variables, method=self.optimizer,
                                 options=dict(maxiter=1000, disp=False), step_callback=hist)
                    print("")
                except KeyboardInterrupt:
                    pass  # TOOD: Come up with something better than just pass...
            else:
                raise NotImplementedError(f"I don't know {self.optimizer}")

        run_optimisation()

        # Store results
        self.trained_parameters = gpflow.utilities.read_values(model)
        self.train_objective_hist = (hist.n_iters, hist.log_likelihoods)


#
#
# Sparse runs
init_Z_runs = {}
for init_Z_method in init_Z_methods:
    settings_for_runs = [{"model_class": "SGPR", "M": M, "init_Z_method": init_Z_method, **dataset_custom_settings}
                         for M in Ms]
    init_Z_runs[init_Z_method] = []
    for run_settings in settings_for_runs:
        run = FullbatchUciInducingOptExperiment(**{**common_run_settings, **run_settings},
                                                initial_parameters=model_parameters)
        run.setup_model()
        run.init_params()
        print_post_run(run)
        init_Z_runs[init_Z_method].append(run)

#
#
# Bound optimisation
settings_for_runs = [{"model_class": "SGPR", "M": M, "init_Z_method": "greedy-trace", **dataset_custom_settings}
                     for M in Ms]
init_Z_runs["optimise"] = []
upper_runs = []
# for optimise_objective in ["upper", "lower"]:  # Optimising the upper bound makes hardly any difference
for optimise_objective in ["lower"]:
    for run_settings in settings_for_runs:
        run = FullbatchUciInducingOptExperiment(**{**common_run_settings, **run_settings},
                                                initial_parameters=model_parameters,
                                                optimise_objective=optimise_objective)
        run.cached_run()
        print_post_run(run)

        if optimise_objective == "lower":
            init_Z_runs["optimise"].append(run)
        elif optimise_objective == "upper":
            upper_runs.append(run)

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

_, ax = plt.subplots()
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

_, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.plot(Ms, init_Z_rmses[method], label=method)
ax.axhline(full_rmse, label="full GP", linestyle='--')
ax.axhline(l_rmse, label="linear", linestyle='-.')
ax.axhline(m_rmse, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("rmse")

_, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.plot(Ms, init_Z_nlpps[method], label=method)
ax.axhline(full_nlpp, label="full GP", linestyle='--')
ax.axhline(l_nlpp, label="linear", linestyle='-.')
ax.axhline(m_nlpp, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("nlpp")

plt.show()
