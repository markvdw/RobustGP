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

from robustgp_experiments.utils import baselines

import matplotlib
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
font = {'family': 'cmr10', 'size': 24}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
color_dict = {"Uniform":'#999999',"Kmeans": '#ff7f00', "Greedy Conditional Variance":'#4daf4a',"gradient":"#377eb8",
            "RLS":'#a65628', "M-DPP MCMC": '#984ea3'}
name_dict = {"Kmeans": "K-means", "Uniform":"Uniform", "Greedy Conditional Variance":"Greedy var.", "Sample Conditional Variance":"Sample var.","gradient":"Gradient", "RLS":"RLS", "M-DPP MCMC": "M-DPP MCMC"}
plot_title_dict = {"Wilson_energy":"Energy", "Wilson_elevators":"Elevators","Naval_noisy":"Naval with Noise"}
methods_to_ignore = ["Sample Conditional Variance"]
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
                try:
                    elbo, upper, rmse, nlpp = jug.task.value(result)
                except:
                    continue
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
        Naval_noisy=dict(xlim=(10,200),elbo_only_ylim=(37300, 37900), elbo_ylim=(36e3, 43500), nlpp_ylim=(-3.6, -2.1),
                         rmse_ylim=(.0065, .018), include_mean=False, include_linear=False),
        Wilson_energy=dict(xlim=(10,200),elbo_only_ylim=(800, 1050), elbo_ylim=(800, 1350), nlpp_ylim=(-1.7, -.6),
                           rmse_ylim=(.046, .09), include_mean=False, include_linear=False),
        Wilson_elevators=dict(xlim=(0,5000),elbo_only_ylim=(-8500, -6000), elbo_ylim=(-8000, 1000), nlpp_ylim=(.375, .45),
                              rmse_ylim=(.3515, .37), include_mean=False, include_linear=False)
    ).get(dataset, dict(elbo_only_y_lim=None, elbo_ylim=None, include_linear=True, nlpp_ylim=None,
                        rmse_ylim=None))
    l_elbo, l_rmse, l_nlpp = linear_baselines[dataset]
    m_elbo, m_rmse, m_nlpp = mean_baselines[dataset]
    fig, ax = plt.subplots()
    for method in init_Z_runs[dataset].keys():
        if method in methods_to_ignore:
            continue
        l, = ax.plot(init_Z_Ms[dataset][method],init_Z_elbos[dataset][method]["Median"], label=name_dict[method],
                     color=color_dict[method])
        ax.plot(init_Z_Ms[dataset][method], init_Z_uppers[dataset][method]["Median"], label="_nolegend_",
                color=l.get_color(), linestyle=(0, (3, 1, 1, 1, 1, 1)))
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_elbos[dataset][method]["20 pct"],
                        init_Z_elbos[dataset][method]["80 pct"], color=l.get_color(), alpha=.2)
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_uppers[dataset][method]["20 pct"],
                        init_Z_uppers[dataset][method]["80 pct"], color=l.get_color(), alpha=.2,
                        hatch='/', label='_nolegend_')
    ax.axhline(baseline_lmls[dataset], label='Full GP', linestyle="--",color='k')
    if dataset_plot_settings["include_linear"]:
        ax.axhline(l_elbo, label='Linear', linestyle='-.',color='k')
        ax.axhline(m_elbo, label='Mean', linestyle=':',color='k')
    #ax.legend(loc="upper left")
    ax.set_xlabel("M")
    ax.set_ylabel("ELBO")
    ax.set_title(plot_title_dict[dataset])
    ax.set_xlim(dataset_plot_settings["xlim"])
    ax.set_ylim(dataset_plot_settings["elbo_ylim"])
    plt.tight_layout()
    fig.savefig(f"./figures/fixedhyp-{dataset}-elbo.pdf")

    fig, ax = plt.subplots()
    for method in init_Z_runs[dataset].keys():
        if method in methods_to_ignore:
            continue
        l, = ax.plot(init_Z_Ms[dataset][method], init_Z_rmses[dataset][method]["Median"], label=name_dict[method],color=color_dict[method])
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_rmses[dataset][method]["20 pct"],
                        init_Z_rmses[dataset][method]["80 pct"], color=l.get_color(),alpha=.2)
    ax.axhline(full_rmses[dataset], label="Full GP", linestyle='--',color='k')
    if dataset_plot_settings["include_linear"]:    
        ax.axhline(l_rmse, label="Linear", linestyle='-.',color='k')
        ax.axhline(m_rmse, label="Mean", linestyle=':',color='k')
    #ax.legend()
    ax.set_xlabel("M")
    ax.set_ylabel("RMSE")
    #ax.set_title(plot_title_dict[dataset])
    ax.set_xlim(dataset_plot_settings["xlim"])
    ax.set_ylim(dataset_plot_settings["rmse_ylim"])
    plt.tight_layout()
    fig.savefig(f"./figures/fixedhyp-{dataset}-rmse.pdf")

    fig, ax = plt.subplots()
    for method in init_Z_runs[dataset].keys():
        if method in methods_to_ignore:
            continue
        l,=ax.plot(init_Z_Ms[dataset][method], init_Z_nlpps[dataset][method]["Median"], label=name_dict[method],color=color_dict[method])
        ax.fill_between(init_Z_Ms[dataset][method], init_Z_nlpps[dataset][method]["20 pct"],
                        init_Z_nlpps[dataset][method]["80 pct"], color=l.get_color(),alpha=.2)
    ax.axhline(full_nlpps[dataset], label="Full GP", linestyle='--', color='k')
    if dataset_plot_settings["include_linear"]:
        ax.axhline(l_nlpp, label="Linear", linestyle='-.',color='k')
        ax.axhline(m_nlpp, label="Mean", linestyle=':',color='k')
    #ax.legend()
    ax.set_xlabel("M")
    ax.set_ylabel("NLPD")
    ax.set_xlim(dataset_plot_settings["xlim"])
    ax.set_ylim(dataset_plot_settings["nlpp_ylim"])
    #ax.set_title(plot_title_dict[dataset])
    plt.tight_layout()
    fig.savefig(f"./figures/fixedhyp-{dataset}-nlpp.pdf")

plt.show()


