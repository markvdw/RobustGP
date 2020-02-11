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

from inducing_experiments.utils import ExperimentRecord

gpflow.config.set_default_positive_minimum(1e-5)

# %% [markdown]
# # Kin40k

# %% {"tags": ["parameters"]}
MAXITER = 6000

experiment_name = "which-optimiser"
dataset_name = "Wilson_elevators"

# %%
experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"

basic_run_settings_list = [
    # {"model_class": "GPR"},
    {"model_class": "SGPR", "M": 100, "fixed_Z": True},
    {"model_class": "SGPR", "M": 100, "fixed_Z": True, "optimizer": "bfgs"},
    {"model_class": "SGPR", "M": 200, "fixed_Z": True},
    # {"model_class": "SGPR", "M": 200, "fixed_Z": True, "optimizer": "bfgs"},  # CholeskyError
    {"model_class": "SGPR", "M": 200, "fixed_Z": True, "optimizer": "bfgs", "lengthscale_transform": "constrained"},
]

common_params = {"storage_path": experiment_storage_path, "dataset_name": dataset_name}
baseline_runs = [ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name, **basic_run_settings)
                 for basic_run_settings in basic_run_settings_list]

[r.cached_run(MAXITER) for r in baseline_runs]
# [plt.plot(*r.train_objective_hist) for r in baseline_runs]
