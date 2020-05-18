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
import numpy as np

from inducing_init_experiments.utils import baselines

jug.init("jug_init_inducing_fixedhyp.py", "jug_init_inducing_fixedhyp.jugdata")
from jug_init_inducing_fixedhyp import (
    init_Z_runs, init_Z_task_results, baseline_exps, full_rmses, full_nlpps, baseline_lmls, Ms, dataset_names
)
#
#
# Evaluation
init_Z_rmses = {}
init_Z_nlpps = {}
init_Z_elbos = {}
init_Z_uppers = {}
init_Z_Ms = {}
mean_baselines = {}
linear_baselines = {}
for dataset in dataset_names:
    init_Z_rmses[dataset] = {}
    init_Z_nlpps[dataset] = {}
    init_Z_elbos[dataset] = {}
    init_Z_uppers[dataset] = {}
    init_Z_Ms[dataset] = {}

    for init_Z_method in init_Z_runs[dataset].keys():
        init_Z_rmses[dataset][init_Z_method] = dict()
        init_Z_nlpps[dataset][init_Z_method] = dict()
        init_Z_elbos[dataset][init_Z_method] = dict()
        init_Z_uppers[dataset][init_Z_method] = dict()
        init_Z_Ms[dataset][init_Z_method] = []
        for stat in ["Means", "Standard dev.", "Sample std.", "Median", "80 pct", "20 pct"]:
            for metric in [init_Z_rmses,init_Z_nlpps, init_Z_elbos, init_Z_uppers]:
                metric[dataset][init_Z_method][stat] = []
        for M in init_Z_task_results[dataset][init_Z_method].keys():
            init_Z_Ms[dataset][init_Z_method].append(int(M))
            init_Z_rmses[dataset][init_Z_method][M] = []
            init_Z_nlpps[dataset][init_Z_method][M] = []
            init_Z_elbos[dataset][init_Z_method][M] = []
            init_Z_uppers[dataset][init_Z_method][M] = []
            for result in init_Z_task_results[dataset][init_Z_method][M]:
                elbo, upper, rmse, nlpp = jug.task.value(result)
                init_Z_elbos[dataset][str(init_Z_method)][M].append(elbo)
                init_Z_uppers[dataset][str(init_Z_method)][M].append(upper)
                init_Z_rmses[dataset][str(init_Z_method)][M].append(rmse)
                init_Z_nlpps[dataset][str(init_Z_method)][M].append(nlpp)
            for metric in [init_Z_rmses, init_Z_nlpps, init_Z_elbos, init_Z_uppers]:
                metric[dataset][init_Z_method]["Means"].append(np.mean(metric[dataset][init_Z_method][M]))
                metric[dataset][init_Z_method]["Standard dev."].append(np.std(metric[dataset][init_Z_method][M]))
                metric[dataset][init_Z_method]["Sample std."].append(np.std(metric[dataset][init_Z_method][M]) /
                                                            np.sqrt((len(metric[dataset][init_Z_method][M])-1)))
                metric[dataset][init_Z_method]["Median"].append(np.median(metric[dataset][init_Z_method][M]))
                metric[dataset][init_Z_method]["20 pct"].append(np.percentile(metric[dataset][init_Z_method][M],20))
                metric[dataset][init_Z_method]["80 pct"].append(np.percentile(metric[dataset][init_Z_method][M],80))


    baseline_exp = baseline_exps[dataset]
    mean_baselines[dataset] = baselines.meanpred_baseline(None, baseline_exp.Y_train, None, baseline_exp.Y_test)
    linear_baselines[dataset] = baselines.linear_baseline(baseline_exp.X_train, baseline_exp.Y_train,
                                                          baseline_exp.X_test, baseline_exp.Y_test)

#
# Plotting
for dataset in dataset_names:
    dataset_plot_settings = dict(
        Naval_noisy=dict(elbo_ylim=(-20e3, 45e3))
        Wilson_energy = dict(elbo_ylim=(-2000, 1500))
    ).get(dataset, dict(elbo_ylim=None))
    l_elbo, l_rmse, l_nlpp = linear_baselines[dataset]
    m_elbo, m_rmse, m_nlpp = mean_baselines[dataset]
    fig, ax = plt.subplots()
    methods = init_Z_runs[dataset].keys()
    for method in methods:
        l, = ax.plot(init_Z_Ms[dataset][method],init_Z_elbos[dataset][method]["Median"], label=f"{method} elbo")
        ax.plot(init_Z_Ms[dataset][method], init_Z_uppers[dataset][method]["Median"], label=f"{method} upper",
                color=l.get_color(), linestyle=(0, (3, 1, 1, 1, 1, 1)))
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_elbos[dataset][method]["20 pct"],
                        init_Z_elbos[dataset][method]["80 pct"], color=l.get_color(),alpha=.2)
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_uppers[dataset][method]["20 pct"],
                        init_Z_uppers[dataset][method]["80 pct"], color=l.get_color(),alpha=.2,
                        hatch='/')
    ax.axhline(baseline_lmls[dataset], label='full GP', linestyle="--")
    ax.axhline(l_elbo, label='linear', linestyle='-.')
    ax.axhline(m_elbo, label='mean', linestyle=':')
    ax.legend()
    ax.set_xlabel("M")
    ax.set_ylabel("elbo")
    ax.set_ylim(dataset_plot_settings["elbo_ylim"])
    fig.savefig(f"./figures/fixedhyp-{dataset}-elbo.png")

    fig, ax = plt.subplots()
    for method in methods:
        ax.plot(init_Z_Ms[dataset][method], init_Z_rmse[dataset][method]["Median"], label=f"{method} rmse",
                color=l.get_color(), linestyle=(0, (3, 1, 1, 1, 1, 1)))
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_rmse[dataset][method]["20 pct"],
                        init_Z_rmse[dataset][method]["80 pct"], color=l.get_color(),alpha=.2)
    ax.axhline(full_rmses[dataset], label="full GP", linestyle='--')
    ax.axhline(l_rmse, label="linear", linestyle='-.')
    ax.axhline(m_rmse, label="mean", linestyle=':')
    ax.legend()
    ax.set_xlabel("M")
    ax.set_ylabel("rmse")
    fig.savefig(f"./figures/fixedhyp-{dataset}-rmse.png")

    fig, ax = plt.subplots()
    for method in methods:
        ax.plot(init_Z_Ms[dataset][method], init_Z_nlpps[dataset][method]["Median"], label=f"{method} nlpp",
                color=l.get_color(), linestyle=(0, (3, 1, 1, 1, 1, 1)))
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_nlpps[dataset][method]["20 pct"],
                        init_Z_nlpps[dataset][method]["80 pct"], color=l.get_color(),alpha=.2)
    ax.axhline(full_nlpps[dataset], label="full GP", linestyle='--')
    ax.axhline(l_nlpp, label="linear", linestyle='-.')
    ax.axhline(m_nlpp, label="mean", linestyle=':')
    ax.legend()
    ax.set_xlabel("M")
    ax.set_ylabel("nlpp")
    fig.savefig(f"./figures/fixedhyp-{dataset}-nlpp.png")

plt.show()
