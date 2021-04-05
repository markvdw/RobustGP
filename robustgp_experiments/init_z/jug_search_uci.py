# # Search-UCI
# Search UCI datasets for datasets for which the sparse approximation effectively converges. I.e. marglik bound doesn't
# increase much as we add more inducing points.

import jug
import numpy as np

import gpflow
import robustgp as inducing_init
from robustgp_experiments.init_z.utils import uci_train_settings, good_datasets, print_post_run
from robustgp_experiments.utils import FullbatchUciExperiment

gpflow.config.set_default_jitter(1e-8)
gpflow.config.set_default_positive_minimum(1.0e-5)

MAXITER = 1000

experiment_name = "search-uci"
dataset_names = ["Wilson_energy", "Wilson_autompg", "Wilson_concrete", "Wilson_airfoil", "Wilson_servo",
                 "Wilson_concreteslump"]
# dataset_names = ["Wilson_stock", "Wilson_energy", "Pendulum_noisy", "Wilson_pendulum", "Wilson_concrete",
#                  "Wilson_airfoil", "Wilson_wine", "Naval_noisy", "Naval", "Wilson_gas", "Wilson_skillcraft",
#                  "Wilson_sml", "Wilson_parkinsons", "Parkinsons_noisy", "Power", "Wilson_pol", "Wilson_elevators",
#                  "Wilson_bike", "Wilson_kin40k", "Wilson_protein", "Wilson_tamielectric"]
# dataset_names = good_datasets

Z_init_method = inducing_init.ConditionalVariance(sample=True)

baseline_exps = {}
baseline_results = {}
sparse_exps = {}
sparse_task_results = {}


def get_settings(dataset_name):
    experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"
    Ms, dataset_custom_settings = uci_train_settings[dataset_name]
    common_run_settings = dict(storage_path=experiment_storage_path, dataset_name=dataset_name,
                               training_procedure="reinit_Z")
    return experiment_storage_path, Ms, common_run_settings, dataset_custom_settings


@jug.TaskGenerator
def full_cached_run(exp):
    exp.cached_run()
    print_post_run(exp)
    return exp.model.robust_maximum_log_likelihood_objective()


@jug.TaskGenerator
def sparse_cached_run(exp):
    exp.cached_run()
    print_post_run(exp)
    # lml = exp.model.log_marginal_likelihood().numpy() if exp.model_class == "GPR" else exp.model.elbo().numpy()
    lml = exp.model.robust_maximum_log_likelihood_objective(restore_jitter=False).numpy()
    upper = exp.model.upper_bound().numpy()
    # rmse = np.mean((exp.model.predict_f(exp.X_test)[0].numpy() - exp.Y_test) ** 2.0) ** 0.5
    # nlpp = -np.mean(exp.model.predict_log_density((exp.X_test, exp.Y_test)))
    return lml, upper, None, None


for dataset_name in dataset_names:
    experiment_storage_path, Ms, common_run_settings, dataset_custom_settings = get_settings(dataset_name)

    #
    #
    # Baseline runs
    gpr_exp = FullbatchUciExperiment(**{**common_run_settings, **dataset_custom_settings, "model_class": "GPR",
                                        "training_procedure": "joint"})
    gpr_exp.load_data()
    if len(gpr_exp.X_train) <= 30000:
        print("Baseline run...")
        result = full_cached_run(gpr_exp)
    else:
        print(f"{dataset_name}: Skipping baseline run... N={len(gpr_exp.X_train)}.")
        result = np.nan

    baseline_exps[dataset_name] = gpr_exp
    baseline_results[dataset_name] = result

    #
    #
    # Sparse runs -- We're trying an "optimal" training procedure
    greedy_init_settings_list = [
        {"model_class": "SGPR", "M": M, "init_Z_method": Z_init_method, **dataset_custom_settings}
        for M in Ms]
    sparse_exps[dataset_name] = []
    sparse_task_results[dataset_name] = []
    for run_settings in greedy_init_settings_list:
        exp = FullbatchUciExperiment(**{**common_run_settings, **run_settings})
        result = sparse_cached_run(exp)
        sparse_exps[dataset_name].append(exp)
        sparse_task_results[dataset_name].append(result)
