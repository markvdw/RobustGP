import jug.task
import matplotlib.pyplot as plt
import numpy as np

from inducing_init_experiments.utils import baselines


jug.init("jug_init_inducing_opt.py", "jug_init_inducing_opt.jugdata")
from jug_init_inducing_opt import (
    init_Z_runs, init_Z_task_results, baseline_exp, full_rmse, full_nlpp, baseline_lml, Ms, dataset_name
)

# Evaluation
init_Z_rmses = {}
init_Z_nlpps = {}
init_Z_elbos = {}
init_Z_uppers = {}
for init_Z_method in init_Z_runs.keys():
    init_Z_rmses[init_Z_method] = dict()
    init_Z_nlpps[init_Z_method] = dict()
    init_Z_elbos[init_Z_method] = dict()
    init_Z_uppers[init_Z_method] = dict()
    for stat in ["Means", "Standard dev.", "Sample std."]:
        for dictionary in [init_Z_rmses, init_Z_nlpps, init_Z_elbos, init_Z_uppers]:
            dictionary[init_Z_method][stat] = []
    for M in Ms:
        init_Z_rmses[init_Z_method][str(M)] = []
        init_Z_nlpps[init_Z_method][str(M)] = []
        init_Z_elbos[init_Z_method][str(M)] = []
        init_Z_uppers[init_Z_method][str(M)] = []
        for result in init_Z_task_results[init_Z_method][str(M)]:
            elbo, upper, rmse, nlpp = jug.task.value(result)
            init_Z_elbos[str(init_Z_method)][str(M)].append(elbo)
            init_Z_uppers[str(init_Z_method)][str(M)].append(upper)
            init_Z_rmses[str(init_Z_method)][str(M)].append(rmse)
            init_Z_nlpps[str(init_Z_method)][str(M)].append(nlpp)
        for dictionary in [init_Z_rmses, init_Z_nlpps, init_Z_elbos, init_Z_uppers]:
            dictionary[init_Z_method]["Means"].append(np.mean(dictionary[str(init_Z_method)][str(M)]))
            dictionary[init_Z_method]["Standard dev."].append(np.std(dictionary[str(init_Z_method)][str(M)]))
            dictionary[init_Z_method]["Sample std."].append(np.std(dictionary[str(init_Z_method)][str(M)])/
                                                            np.sqrt((len(dictionary[str(init_Z_method)][str(M)])-1)))


m_elbo, m_rmse, m_nlpp = baselines.meanpred_baseline(None, baseline_exp.Y_train, None, baseline_exp.Y_test)
l_elbo, l_rmse, l_nlpp = baselines.linear_baseline(baseline_exp.X_train, baseline_exp.Y_train,
                                                   baseline_exp.X_test, baseline_exp.Y_test)


# Plotting
print(f"gpr rmse: {full_rmse}")
print(f"rmse    : {init_Z_rmses}")
print(f"gpr nlpp: {full_nlpp}")
print(f"nlpp    : {init_Z_nlpps}")
dataset_plot_settings = dict(
    Naval_noisy=dict(elbo_ylim=(-20e3, 45e3))
).get(dataset_name, dict(elbo_ylim=None))

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    l,_,_ = ax.errorbar(Ms, init_Z_elbos[method]["Means"], yerr = init_Z_elbos[method]["Sample std."], label=f"{method} elbo")
    ax.errorbar(Ms, init_Z_uppers[method]["Means"], yerr = init_Z_elbos[method]["Sample std."], label=f"{method} upper", color=l.get_color(), linestyle=(0, (3, 1, 1, 1, 1, 1)))
ax.axhline(baseline_lml, label='full GP', linestyle="--")
ax.axhline(l_elbo, label='linear', linestyle='-.')
ax.axhline(m_elbo, label='mean', linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("elbo")
ax.set_ylim(dataset_plot_settings["elbo_ylim"])
fig.savefig(f"./figures/opthyp-{dataset_name}-elbo.png")

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.errorbar(Ms, init_Z_rmses[method]["Means"], yerr=init_Z_rmses[method]["Sample std."], label=method)
ax.axhline(full_rmse, label="full GP", linestyle='--')
ax.axhline(l_rmse, label="linear", linestyle='-.')
ax.axhline(m_rmse, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("rmse")
fig.savefig(f"./figures/opthyp-{dataset_name}-rmse.png")

fig, ax = plt.subplots()
for method in init_Z_runs.keys():
    ax.errorbar(Ms, init_Z_nlpps[method]["Means"], yerr=init_Z_nlpps[method]["Sample std."], label=method)
ax.axhline(full_nlpp, label="full GP", linestyle='--')
ax.axhline(l_nlpp, label="linear", linestyle='-.')
ax.axhline(m_nlpp, label="mean", linestyle=':')
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("nlpp")
fig.savefig(f"./figures/opthyp-{dataset_name}-nlpp.png")

plt.show()