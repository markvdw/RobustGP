# # Inducing point initialisation while optimising hypers
# Assess how well inducing point initialisation works, in conjunction with optimising the hyperparameters.
# Here, we compare:
#  - Fixed Z,       initialised with the baseline kernel hyperparameters
#  - EM "reinit" Z, initialised with the baseline kernel hyperparameters
# In all cases, the Z are initialised using default kernel parameters, not the ones from the baseline. This is to ensure
# that we do not use information when initialising Z that isn't accessable when running a new dataset.
#
# Local optima are a bit annoying when "cold" initialising. They make the plot appear non-smooth. So we initialise at


import gpflow
import jug
import numpy as np

import robustgp
from robustgp_experiments.init_z.utils import uci_train_settings, print_post_run
from robustgp_experiments.utils import FullbatchUciExperiment

# Settings
dataset_names = ["Wilson_energy", "Naval_noisy", "Wilson_elevators"]
init_from_baseline = False

uci_train_settings.update(dict(Naval_noisy=([55], {}), Wilson_energy=([65], {}), Wilson_elevators=([1200], {})))

# Setup
gpflow.config.set_default_positive_minimum(1.0e-5)
gpflow.config.set_default_jitter(1e-10)

num_seeds = 3  # For the experiments in the paper we used 10
seeds = np.arange(num_seeds)

init_Z_methods = dict()
init_Z_methods["Kmeans"] = [robustgp.Kmeans(seed=seed) for seed in seeds]
init_Z_methods["Greedy Conditional Variance"] = [robustgp.ConditionalVariance(seed=seed) for seed in seeds]
init_Z_methods["Gradient"] = [robustgp.ConditionalVariance(seed=seed) for seed in seeds]
experiment_name = "opt-inducing"


def compute_model_stats(exp):
    elbo = exp.model.robust_maximum_log_likelihood_objective(restore_jitter=False).numpy()
    rmse = np.mean((exp.model.predict_f(exp.X_test)[0].numpy() - exp.Y_test) ** 2.0) ** 0.5
    nlpp = -np.mean(exp.model.predict_log_density((exp.X_test, exp.Y_test)))
    upper = exp.model.upper_bound().numpy()
    hist = exp.train_objective_hist
    return elbo, upper, rmse, nlpp, hist


@jug.TaskGenerator
def run_baseline(baseline_exp):
    baseline_exp.cached_run()
    baseline_lml = baseline_exp.model.robust_maximum_log_likelihood_objective().numpy()
    model_parameters = gpflow.utilities.read_values(baseline_exp.model) if init_from_baseline else {}
    if ".inducing_variable.Z" in model_parameters:
        model_parameters.pop(".inducing_variable.Z")
    full_rmse = (
        np.mean((baseline_exp.model.predict_f(baseline_exp.X_test)[0].numpy() - baseline_exp.Y_test) ** 2.0) ** 0.5
    )
    full_nlpp = -np.mean(baseline_exp.model.predict_log_density((baseline_exp.X_test, baseline_exp.Y_test)))
    return model_parameters, full_rmse, full_nlpp, baseline_lml


baseline_exps = dict()
baseline_tasks = []
for dataset_name in dataset_names:
    # Baseline runs
    print("Baseline exp...", dataset_name)
    experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"
    common_run_settings = dict(storage_path=experiment_storage_path, dataset_name=dataset_name)
    Ms, dataset_custom_settings = uci_train_settings[dataset_name]

    baseline_custom_settings = dict(
        Naval_noisy={
            "model_class": "SGPR",
            "M": 1000,
            "training_procedure": "reinit_Z",
            "init_Z_method": robustgp.ConditionalVariance(sample=False),
            "max_lengthscale": 1000.0,
            "max_variance": 1000.0,
        }
    ).get(
        dataset_name, dict(model_class="GPR", training_procedure="joint", max_lengthscale=1000.0, max_variance=1000.0)
    )
    baseline_exps[dataset_name] = FullbatchUciExperiment(
        **{**common_run_settings, **dataset_custom_settings, **baseline_custom_settings}
    )

    baseline_tasks.append(run_baseline(baseline_exps[dataset_name]))


# Bound optimisation
@jug.TaskGenerator
def run_sparse_opt(exp):
    print(exp)
    exp.cached_run()
    print_post_run(exp)
    elbo, upper, rmse, nlpp, hist = compute_model_stats(exp)
    return elbo, upper, rmse, nlpp, hist


# Sparse runs
init_Z_runs = dict()
init_Z_task_results = dict()
for dataset_name in dataset_names:
    experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"
    common_run_settings = dict(
        storage_path=experiment_storage_path,
        dataset_name=dataset_name,
        max_lengthscale=1001.0,
        max_variance=1001.0,
        training_procedure="reinit_Z",
    )
    Ms, dataset_custom_settings = uci_train_settings[dataset_name]

    init_Z_runs[dataset_name] = {}
    init_Z_task_results[dataset_name] = {}
    for method_name, init_Z_method in init_Z_methods.items():
        training_procedure = "reinit_Z"
        settings_for_runs = [
            {
                "model_class": "SGPR",
                "M": M,
                "init_Z_method": seeded_init_Z_method,
                "base_filename": "opthyp-fixed_Z",
                "initial_parameters": {},
                "training_procedure": training_procedure,
                **dataset_custom_settings,
            }
            for M in Ms
            for seeded_init_Z_method in init_Z_method
        ]
        init_Z_runs[dataset_name][method_name] = dict()
        init_Z_task_results[dataset_name][method_name] = dict()
        for M in Ms:
            init_Z_runs[dataset_name][method_name][str(M)] = []
            init_Z_task_results[dataset_name][method_name][str(M)] = []
        for run_settings in settings_for_runs:
            M = str(run_settings["M"])
            exp = FullbatchUciExperiment(**{**common_run_settings, **run_settings})
            result = run_sparse_opt(exp)
            init_Z_runs[dataset_name][method_name][M].append(exp)
            init_Z_task_results[dataset_name][method_name][M].append(result)

# Optimisation of Z
method_name = "Gradient"
for dataset_name in dataset_names:
    experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"
    common_run_settings = dict(
        storage_path=experiment_storage_path, dataset_name=dataset_name, max_lengthscale=1001.0, max_variance=1001.0
    )
    Ms, dataset_custom_settings = uci_train_settings[dataset_name]
    settings_for_runs = [
        {
            "model_class": "SGPR",
            "M": M,
            "training_procedure": "joint",
            "base_filename": "opthyp-reinit_Z",
            "initial_parameters": {},
            **dataset_custom_settings,
        }
        for M in Ms
    ]

    init_Z_runs[dataset_name][method_name] = dict()
    init_Z_task_results[dataset_name][method_name] = dict()
    for M in Ms:
        init_Z_runs[dataset_name][method_name][str(M)] = []
        init_Z_task_results[dataset_name][method_name][str(M)] = []

    for seed in seeds:
        init_Z_method = robustgp.ConditionalVariance(seed=seed, sample=False)
        for run_settings in settings_for_runs:
            exp = FullbatchUciExperiment(**{**common_run_settings, **run_settings, "init_Z_method": init_Z_method})
            result = run_sparse_opt(exp)
            M = str(run_settings["M"])
            init_Z_runs[dataset_name][method_name][M].append(exp)
            init_Z_task_results[dataset_name][method_name][M].append(result)


all_model_parameters = dict()
full_rmses = dict()
full_nlpps = dict()
baseline_lmls = dict()
# Put trained values in variables
for dataset_name in dataset_names:
    (
        all_model_parameters[dataset_name],
        full_rmses[dataset_name],
        full_nlpps[dataset_name],
        baseline_lmls[dataset_name],
    ) = jug.bvalue(run_baseline(baseline_exps[dataset_name]))
