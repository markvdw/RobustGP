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

import matplotlib.pyplot as plt
import numpy as np
from inducing_experiments.utils import ExperimentRecord, baselines

import gpflow

gpflow.config.set_default_positive_minimum(1.0e-5)

# %% [markdown]
# # Kin40k

# %% {"tags": ["parameters"]}
MAXITER = 6000

experiment_name = "init-inducing"
dataset_name = "Wilson_elevators"

# %%
experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"
common_run_settings = {"optimizer": "l-bfgs-b", "normalise_on_training": True}  # True is the old way of doing things, and the more correct one
# common_run_settings = {"optimizer": "bfgs", "lengthscale_transform": "constrained"}

Ms = dict(
    Wilson_elevators=[100, 200, 500, 1000, 1500, 1750, 2000],
    # Wilson_elevators=[100, 200, 500, 1000],
    Wilson_stock=[10, 50, 100, 200, 500],
    Wilson_housing=[100, 200, 300, 400, 500]
)[dataset_name]


def print_post_run(run):
    print("")
    std_ratio = (run.model.kernel.variance.numpy() / run.model.likelihood.variance.numpy()) ** 0.5
    print(f"(kernel.variance / likelihood.variance)**0.5: {std_ratio}")
    print(run.model.kernel.lengthscales.numpy())
    print("")
    print("")


#
#
# Baseline runs
# basic_run_settings_list = [{"model_class": "SGPR", "M": M, "fixed_Z": True} for M in Ms]
# baseline_runs = [ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name,
#                                   **{**run_settings, **common_run_settings})
#                  for run_settings in basic_run_settings_list]
# for i in range(len(baseline_runs)):
#     # Initialising from the previous solution does not really change the result, it simply speeds things up.
#     # Either way, that would be a question of local optima.
#     baseline_runs[i].cached_run(MAXITER)
#     print_post_run(baseline_runs[i])

#
#
# Greedy initialised runs
greedy_init_settings_list = [{"model_class": "SGPR", "M": M, "fixed_Z": True, "init_Z_method": "greedy-trace"}
                             for M in Ms]
greedy_init_runs = [ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name,
                                     **{**run_settings, **common_run_settings})
                    for run_settings in greedy_init_settings_list]
for i in range(len(greedy_init_runs)):
    # , init_from_model=None if i <= 2 else greedy_init_runs[i - 1].model
    greedy_init_runs[i].cached_run(MAXITER)
    # greedy_init_runs[i]._load_data()
    print_post_run(greedy_init_runs[i])

#
#
# Plotting
greedy_rmses = [np.mean((exp.model.predict_f(exp.X_test)[0].numpy() - exp.Y_test) ** 2.0) ** 0.5
                for exp in greedy_init_runs]
greedy_nlpps = [-np.mean(exp.model.predict_log_density((exp.X_test, exp.Y_test)))
                for exp in greedy_init_runs]
print(greedy_rmses)
print(greedy_nlpps)

m_elbo, m_rmse, m_nlpp = baselines.meanpred_baseline(None, greedy_init_runs[0].Y_train, None, greedy_init_runs[0].Y_test)
l_elbo, l_rmse, l_nlpp = baselines.linear_baseline(greedy_init_runs[0].X_train, greedy_init_runs[0].Y_train,
                                                   greedy_init_runs[0].X_test, greedy_init_runs[0].Y_test)

_, ax = plt.subplots()
# plt.plot([r.M for r in baseline_runs], [r.model.elbo().numpy() for r in baseline_runs], '-x')
ax.plot([r.M for r in greedy_init_runs], [r.model.elbo().numpy() for r in greedy_init_runs], '-x')
ax.plot([r.M for r in greedy_init_runs], [r.model.upper_bound().numpy() for r in greedy_init_runs], '-x')
ax.set_xlabel("M")
ax.set_ylabel("elbo")

_, ax = plt.subplots()
# [plt.plot(*r.train_objective_hist) for r in baseline_runs]
[ax.plot(*r.train_objective_hist) for r in greedy_init_runs]
ax.axhline(-m_elbo, linestyle=':')
ax.axhline(-l_elbo, linestyle='-.')
ax.set_xlabel("iters")
ax.set_ylabel("elbo")

_, ax = plt.subplots()
ax.plot(Ms, greedy_rmses)
ax.axhline(m_rmse, linestyle=':')
ax.axhline(l_rmse, linestyle='-.')
ax.set_xlabel("M")
ax.set_ylabel("rmse")

_, ax = plt.subplots()
ax.plot(Ms, greedy_nlpps)
ax.axhline(m_nlpp, linestyle=':')
ax.axhline(l_nlpp, linestyle='-.')
ax.set_xlabel("M")
ax.set_ylabel("nlpp")

plt.show()

# Compare convergence of normalise_on_training on and off
