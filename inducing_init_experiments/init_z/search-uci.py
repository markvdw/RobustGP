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

# %% [markdown]
# # Search-UCI
# Search UCI datasets for datasets for which the sparse approximation effectively converges. I.e. marglik bound doesn't
# increase much as we add more inducing points.

import gpflow
import matplotlib.pyplot as plt
import numpy as np
from inducing_experiments.utils import baselines, FullbatchUciExperiment

gpflow.config.set_default_positive_minimum(1.0e-5)

# %% {"tags": ["parameters"]}
MAXITER = 1000

experiment_name = "init-inducing"
dataset_name = "Wilson_pol"

# %%
experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"

Ms, dataset_custom_settings = dict(
    Wilson_pol=([100, 200, 500, 1000, 2000], {}),
    Naval_noisy=([10, 20, 50, 100, 200, 500], {}),  # Very sparse solution exists
    Naval=([10, 20, 50, 100, 200, 500], {}),  # Very sparse solution exists
    Power=([100, 200, 500, 1000, 2000], {}),  # Step function in it?
    Kin8mn=([100, 200, 500, 1000, 2000], {}),  # Can't download
    Wilson_parkinsons=([100, 200, 500, 1000], {"max_lengthscale": 10.0}),  # Cholesky errors
    Wilson_sml=([100, 200, 500, 1000, 2000, 3000, 3500], {}),  # Mostly linear, but with benefit of nonlinear
    # Didn't get SE+Lin working, probably local optimum
    # Wilson_skillcraft=([10, 20, 50, 100, 200, 500], {"kernel_name": "SquaredExponentialLinear"}),
    Wilson_skillcraft=([10, 20, 50, 100, 200, 500, 1000], {}),  # Mostly linear, but with benefit of nonlinear
    Wilson_gas=([100, 200, 500, 1000, 1300], {}),
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


common_run_settings = dict(storage_path=experiment_storage_path, dataset_name=dataset_name,
                           training_procedure="fixed_Z")

#
#
# Baseline runs
gpr_exp = FullbatchUciExperiment(**{**common_run_settings, **dataset_custom_settings, "model_class": "GPR"})
gpr_exp.load_data()
run_gpr = len(gpr_exp.X_train) <= 5000
if run_gpr:
    print("Baseline run...")
    gpr_exp.cached_run()
else:
    print("Skipping baseline run...")

#
#
# Greedy initialised runs
greedy_init_settings_list = [{"model_class": "SGPR", "M": M, "init_Z_method": "greedy-trace", **dataset_custom_settings}
                             for M in Ms]
greedy_init_runs = [FullbatchUciExperiment(**{**common_run_settings, **run_settings})
                    for run_settings in greedy_init_settings_list]
for i in range(len(greedy_init_runs)):
    greedy_init_runs[i].cached_run()
    print_post_run(greedy_init_runs[i])

#
#
# Plotting
greedy_rmses = [np.mean((exp.model.predict_f(exp.X_test)[0].numpy() - exp.Y_test) ** 2.0) ** 0.5
                for exp in greedy_init_runs]
greedy_nlpps = [-np.mean(exp.model.predict_log_density((exp.X_test, exp.Y_test)))
                for exp in greedy_init_runs]

if run_gpr:
    full_rmse = np.mean((gpr_exp.model.predict_f(gpr_exp.X_test)[0].numpy() - gpr_exp.Y_test) ** 2.0) ** 0.5
    full_nlpp = -np.mean(gpr_exp.model.predict_log_density((gpr_exp.X_test, gpr_exp.Y_test)))
else:
    full_rmse, full_nlpp = np.nan, np.nan
print(f"gpr rmse: {full_rmse}")
print(f"rmse    : {greedy_rmses}")
print(f"gpr nlpp: {full_nlpp}")
print(f"nlpp    : {greedy_nlpps}")

m_elbo, m_rmse, m_nlpp = baselines.meanpred_baseline(None, greedy_init_runs[0].Y_train, None,
                                                     greedy_init_runs[0].Y_test)
l_elbo, l_rmse, l_nlpp = baselines.linear_baseline(greedy_init_runs[0].X_train, greedy_init_runs[0].Y_train,
                                                   greedy_init_runs[0].X_test, greedy_init_runs[0].Y_test)

_, ax = plt.subplots()
# plt.plot([r.M for r in baseline_runs], [r.model.elbo().numpy() for r in baseline_runs], '-x')
ax.plot([r.M for r in greedy_init_runs], [r.model.elbo().numpy() for r in greedy_init_runs], '-x')
ax.plot([r.M for r in greedy_init_runs], [r.model.upper_bound().numpy() for r in greedy_init_runs], '-x')
if run_gpr:
    ax.axhline(gpr_exp.model.log_marginal_likelihood().numpy(), linestyle="--")
ax.set_xlabel("M")
ax.set_ylabel("elbo")

_, ax = plt.subplots()
# [plt.plot(*r.train_objective_hist) for r in baseline_runs]
[ax.plot(*r.train_objective_hist) for r in greedy_init_runs]
if run_gpr:
    ax.axhline(-gpr_exp.model.log_marginal_likelihood().numpy(), linestyle="--")
ax.axhline(-m_elbo, linestyle=':')
ax.axhline(-l_elbo, linestyle='-.')
ax.set_xlabel("iters")
ax.set_ylabel("elbo")

_, ax = plt.subplots()
ax.plot(Ms, greedy_rmses)
ax.axhline(m_rmse, linestyle=':')
ax.axhline(l_rmse, linestyle='-.')
ax.axhline(full_rmse, linestyle='--')
ax.set_xlabel("M")
ax.set_ylabel("rmse")

_, ax = plt.subplots()
ax.plot(Ms, greedy_nlpps)
ax.axhline(m_nlpp, linestyle=':')
ax.axhline(l_nlpp, linestyle='-.')
ax.axhline(full_nlpp, linestyle='--')
ax.set_xlabel("M")
ax.set_ylabel("nlpp")

plt.show()
