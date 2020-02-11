# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# ../create_report.sh jpt-test.py

import gpflow
import matplotlib.pyplot as plt

from inducing_experiments.utils import ExperimentRecord

gpflow.config.set_default_positive_minimum(1e-5)

# %% [markdown]
# # Kin40k

# %% {"tags": ["parameters"]}
MAXITER = 6000

experiment_name = "regression"
dataset_name = "Wilson_elevators"

# %%
experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"
common_run_settings = {"optimizer": "l-bfgs-b"}
# common_run_settings = {"optimizer": "bfgs", "lengthscale_transform": "constrained"}

basic_run_settings_list = [
    # {"model_class": "GPR"},
    {"model_class": "SGPR", "M": 100, "fixed_Z": True},
    {"model_class": "SGPR", "M": 200, "fixed_Z": True},
    {"model_class": "SGPR", "M": 500, "fixed_Z": True},
    {"model_class": "SGPR", "M": 1000, "fixed_Z": True},
    # {"model_class": "SGPR", "M": 2000, "fixed_Z": True},
    # {"model_class": "SGPR", "M": 5000, "fixed_Z": True},
]

common_params = {"storage_path": experiment_storage_path, "dataset_name": dataset_name}
baseline_runs = [ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name,
                                  **{**run_settings, **common_run_settings})
                 for run_settings in basic_run_settings_list]
for i in range(len(baseline_runs)):
    # Initialising from the previous solution does not really change the result, it simply speeds things up.
    # Either way, that would be a question of local optima.
    baseline_runs[i].cached_run(MAXITER, init_from_model=None if i == 0 else baseline_runs[i - 1].model)
    print(baseline_runs[i].model.kernel.lengthscale.numpy())

greedy_init_settings_list = [
    {"model_class": "SGPR", "M": 100, "fixed_Z": True, "init_Z_method": "greedy-trace"},
    {"model_class": "SGPR", "M": 200, "fixed_Z": True, "init_Z_method": "greedy-trace"},
    {"model_class": "SGPR", "M": 500, "fixed_Z": True, "init_Z_method": "greedy-trace"},
    {"model_class": "SGPR", "M": 1000, "fixed_Z": True, "init_Z_method": "greedy-trace"},
    # {"model_class": "SGPR", "M": 2000, "fixed_Z": True, "init_Z_method": "greedy-trace"},
    # {"model_class": "SGPR", "M": 5000, "fixed_Z": True, "init_Z_method": "greedy-trace"},
]
greedy_init_runs = [ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name,
                                     **{**run_settings, **common_run_settings})
                    for run_settings in greedy_init_settings_list]
import numpy as np
np.seterr(all='raise')
for i in range(len(greedy_init_runs)):
    greedy_init_runs[i].cached_run(MAXITER, init_from_model=None if i <= 2 else greedy_init_runs[i - 1].model)
    print(greedy_init_runs[i].model.kernel.lengthscale.numpy())
# [r.cached_run(MAXITER) for r in greedy_init_runs]

plt.plot([r.M for r in baseline_runs], [r.model.log_likelihood().numpy() for r in baseline_runs], '-x')
plt.plot([r.M for r in greedy_init_runs], [r.model.log_likelihood().numpy() for r in greedy_init_runs], '-x')
plt.plot([r.M for r in greedy_init_runs], [r.model.upper_bound().numpy() for r in greedy_init_runs], '-x')
plt.figure()
[plt.plot(*r.train_objective_hist) for r in baseline_runs]
[plt.plot(*r.train_objective_hist) for r in greedy_init_runs]
