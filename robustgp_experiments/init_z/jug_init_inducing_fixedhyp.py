from dataclasses import dataclass
from typing import Optional

import gpflow
import jug
import numpy as np
import tensorflow as tf

import robustgp
from robustgp.utilities import set_trainable
from robustgp_experiments.init_z.utils import print_post_run, uci_train_settings
from robustgp_experiments.utils import FullbatchUciExperiment, LoggerCallback

# Settings
dataset_names = ["Wilson_energy", "Naval_noisy", "Wilson_elevators"]
num_seeds = 3  # Use 10 to replicate experiments from paper
seeds = np.arange(num_seeds)
all_model_parameters = {}

#
#
# Setup
gpflow.config.set_default_positive_minimum(1.0e-5)
gpflow.config.set_default_jitter(1e-10)
init_Z_methods = dict()
init_Z_methods["Uniform"] = [robustgp.UniformSubsample(seed=seed) for seed in seeds]
init_Z_methods["Greedy Conditional Variance"] = [robustgp.ConditionalVariance(seed=seed) for seed in seeds]
# init_Z_methods["Sample Conditional Variance"] = [robustgp.ConditionalVariance(sample=True, seed=seed) for seed in seeds]
init_Z_methods["Kmeans"] = [robustgp.Kmeans(seed=seed) for seed in seeds]
# init_Z_methods["M-DPP MCMC"] = [robustgp.KdppMCMC(seed=seed) for seed in seeds]
# init_Z_methods["RLS"] = [robustgp.RLS(seed=seed) for seed in seeds]

experiment_name = "init-inducing"


uci_train_settings.update(
    dict(
        Naval_noisy=(
            [10, 20, 30, 40, 45, 47, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100, 130, 150, 180, 200, 250, 300, 400, 500],
            {},
        ),  # Very sparse solution exists
    )
)


def get_settings(dataset_name):
    experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"
    Ms, dataset_custom_settings = uci_train_settings[dataset_name]
    common_run_settings = dict(
        storage_path=experiment_storage_path, dataset_name=dataset_name, max_lengthscale=1001.0, max_variance=1001.0
    )
    return experiment_storage_path, Ms, common_run_settings, dataset_custom_settings


#
# Experiment classes
@dataclass
class FullbatchUciInducingOptExperiment(FullbatchUciExperiment):
    optimise_objective: Optional[str] = None  # None | lower | upper

    def run_optimisation(self):
        print(f"Running {str(self)}")

        model = self.model
        set_trainable(model, False)
        set_trainable(model.inducing_variable, True)

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
                    opt.minimize(
                        loss_function,
                        self.model.trainable_variables,
                        method=self.optimizer,
                        options=dict(maxiter=1000, disp=False),
                        step_callback=hist,
                    )
                    print("")
                except KeyboardInterrupt:
                    pass  # TODO: Come up with something better than just pass...
            else:
                raise NotImplementedError(f"I don't know {self.optimizer}")

        run_optimisation()

        # Store results
        self.trained_parameters = gpflow.utilities.read_values(model)
        self.train_objective_hist = (hist.n_iters, hist.log_likelihoods)


def compute_model_stats(exp):
    rmse = np.mean((exp.model.predict_f(exp.X_test)[0].numpy() - exp.Y_test) ** 2.0) ** 0.5
    nlpp = -np.mean(exp.model.predict_log_density((exp.X_test, exp.Y_test)))
    elbo = exp.model.elbo().numpy()
    upper = exp.model.upper_bound().numpy()
    return elbo, upper, rmse, nlpp


@jug.TaskGenerator
def run_baseline(baseline_exp):
    baseline_exp.cached_run()
    if baseline_exp.model_class == "SGPR":
        baseline_lml = baseline_exp.model.elbo().numpy()
    else:
        baseline_lml = baseline_exp.model.log_marginal_likelihood().numpy()
    model_parameters = gpflow.utilities.read_values(baseline_exp.model)
    if ".inducing_variable.Z" in model_parameters:
        model_parameters.pop(".inducing_variable.Z")
    full_rmse = (
        np.mean((baseline_exp.model.predict_f(baseline_exp.X_test)[0].numpy() - baseline_exp.Y_test) ** 2.0) ** 0.5
    )
    full_nlpp = -np.mean(baseline_exp.model.predict_log_density((baseline_exp.X_test, baseline_exp.Y_test)))
    return model_parameters, full_rmse, full_nlpp, baseline_lml


# Baseline exp
baseline_exps = {}
full_rmses = {}
full_nlpps = {}
baseline_lmls = {}
for dataset_name in dataset_names:
    # The baseline is GPR, except for Naval, where we use an SGPR with 1000 inducing variables.
    baseline_custom_settings = dict(
        Naval_noisy={
            "model_class": "SGPR",
            "M": 1000,
            "training_procedure": "reinit_Z",
            "init_Z_method": robustgp.ConditionalVariance(sample=False),
            "max_lengthscale": 1000.0,
            "max_variance": 1000.0,
        }
    ).get(dataset_name, dict(model_class="GPR", max_lengthscale=1000.0, max_variance=1000.0))
    experiment_storage_path, _, common_run_settings, dataset_custom_settings = get_settings(dataset_name)
    baseline_exps[dataset_name] = FullbatchUciExperiment(
        **{**common_run_settings, **dataset_custom_settings, **baseline_custom_settings}
    )
    (
        all_model_parameters[dataset_name],
        full_rmses[dataset_name],
        full_nlpps[dataset_name],
        baseline_lmls[dataset_name],
    ) = jug.bvalue(run_baseline(baseline_exps[dataset_name]))


@jug.TaskGenerator
def run_sparse_init(exp):
    print(exp)
    exp.setup_model()
    exp.init_params()
    print_post_run(exp)
    elbo, upper, rmse, nlpp = compute_model_stats(exp)
    return elbo, upper, rmse, nlpp


# Sparse experiments
init_Z_runs = {}
init_Z_task_results = {}
for dataset_name in dataset_names:
    init_Z_runs[dataset_name] = dict()
    init_Z_task_results[dataset_name] = dict()
    experiment_storage_path, Ms, common_run_settings, dataset_custom_settings = get_settings(dataset_name)
    for method_name, init_Z_method in init_Z_methods.items():
        init_Z_runs[dataset_name][method_name] = dict()
        init_Z_task_results[dataset_name][method_name] = dict()
        for M in Ms:
            init_Z_runs[dataset_name][method_name][str(M)] = []
            init_Z_task_results[dataset_name][method_name][str(M)] = []
        settings_for_runs = [
            {"model_class": "SGPR", "M": M, "init_Z_method": seeded_init_Z_method, **dataset_custom_settings}
            for M in Ms
            for seeded_init_Z_method in init_Z_method
        ]
        for run_settings in settings_for_runs:
            M = str(run_settings["M"])
            # TODO: A better approach would be to put the bvalue at this point, so other jobs could run before the last
            #       full GP finishes.
            exp = FullbatchUciExperiment(
                **{**common_run_settings, **run_settings}, initial_parameters=all_model_parameters[dataset_name]
            )
            result = run_sparse_init(exp)
            init_Z_runs[dataset_name][method_name][M].append(exp)
            init_Z_task_results[dataset_name][method_name][M].append(result)


# (Sparse) Bound optimisation
@jug.TaskGenerator
def run_sparse_opt(exp):
    print(exp)
    exp.cached_run()
    print_post_run(exp)
    elbo, upper, rmse, nlpp = compute_model_stats(exp)
    return elbo, upper, rmse, nlpp


upper_runs = dict()
for dataset_name in dataset_names:
    experiment_storage_path, Ms, common_run_settings, dataset_custom_settings = get_settings(dataset_name)
    settings_for_runs = [
        {
            "model_class": "SGPR",
            "M": M,
            "init_Z_method": robustgp.ConditionalVariance(sample=True, seed=seed),
            **dataset_custom_settings,
        }
        for M in Ms
        for seed in seeds
    ]
    init_Z_runs[dataset_name]["gradient"] = dict()
    init_Z_task_results[dataset_name]["gradient"] = dict()
    upper_runs[dataset_name] = dict()
    for M in Ms:
        init_Z_runs[dataset_name]["gradient"][str(M)] = []
        init_Z_task_results[dataset_name]["gradient"][str(M)] = []
        upper_runs[dataset_name][str(M)] = []
        # for optimise_objective in ["upper", "lower"]:  # Optimising the upper bound makes hardly any difference
    for optimise_objective in ["lower"]:
        for run_settings in settings_for_runs:
            exp = FullbatchUciInducingOptExperiment(
                **{**common_run_settings, **run_settings},
                initial_parameters=all_model_parameters[dataset_name],
                optimise_objective=optimise_objective,
            )
            M = str(run_settings["M"])
            result = run_sparse_opt(exp)
            if optimise_objective == "lower":
                init_Z_runs[dataset_name]["gradient"][str(M)].append(exp)
                init_Z_task_results[dataset_name]["gradient"][str(M)].append(result)
            elif optimise_objective == "upper":
                upper_runs[dataset_name][str(M)].append(exp)
