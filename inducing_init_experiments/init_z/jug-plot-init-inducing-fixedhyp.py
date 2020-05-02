# # Inducing point initlaisation fixed hyperparameters
# Assess how well inducing point initialisation works, with the hyperparameters fixed to the ones found by the full GP.
# This simplifies things, since we only need to run optimisation with the full GP (or a GP with many inducing points).
#
# To parallelise things, we use jug.
#  1. Run `jug execute jug_init_inducing_fixedhyp.py` multiple times to do runs in parallel. Can sync over a shared
#     filesystem.
#  2. Once the above is done, run this script to create the plots.

import jug.task
import matplotlib.pyplot as plt

from inducing_init_experiments.utils import baselines

jug.init("jug_init_inducing_fixedhyp.py", "jug_init_inducing_fixedhyp.jugdata")
from jug_init_inducing_fixedhyp import (
    init_Z_runs, init_Z_task_results, baseline_exp, full_rmse, full_nlpp, baseline_lml, Ms, dataset_name
)

#
#
# Settings
dataset_plot_settings = dict(
    Naval_noisy=dict(elbo_ylim=(-20e3, 45e3))
).get(dataset_name, dict(elbo_ylim=None))

#
#
# Evaluation
init_Z_rmses = {}
init_Z_nlpps = {}
init_Z_elbos = {}
init_Z_uppers = {}
for init_Z_method in init_Z_runs.keys():
    init_Z_rmses[init_Z_method] = []
    init_Z_nlpps[init_Z_method] = []
    init_Z_elbos[init_Z_method] = []
    init_Z_uppers[init_Z_method] = []
    for result in init_Z_task_results[init_Z_method]:
        elbo, upper, rmse, nlpp = jug.task.value(result)
        init_Z_elbos[str(init_Z_method)].append(elbo)
        init_Z_uppers[str(init_Z_method)].append(upper)
        init_Z_rmses[str(init_Z_method)].append(rmse)
        init_Z_nlpps[str(init_Z_method)].append(nlpp)

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
fig.savefig(f"./figures/fixedhyp-{dataset_name}-elbo.png")

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.plot(Ms, init_Z_rmses[method], label=method)
ax.axhline(full_rmse, label="full GP", linestyle='--')
ax.axhline(l_rmse, label="linear", linestyle='-.')
ax.axhline(m_rmse, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("rmse")
fig.savefig(f"./figures/fixedhyp-{dataset_name}-rmse.png")

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.plot(Ms, init_Z_nlpps[method], label=method)
ax.axhline(full_nlpp, label="full GP", linestyle='--')
ax.axhline(l_nlpp, label="linear", linestyle='-.')
ax.axhline(m_nlpp, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("nlpp")
fig.savefig(f"./figures/fixedhyp-{dataset_name}-nlpp.png")

plt.show()
