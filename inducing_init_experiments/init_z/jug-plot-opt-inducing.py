import jug.task
import matplotlib.pyplot as plt
import numpy as np

jug.init("jug_opt_inducing.py", "jug_opt_inducing.jugdata")
from jug_opt_inducing import (init_Z_runs, init_Z_task_results, baseline_lmls)

hists = {}
for dataset in init_Z_runs.keys():
    hists[dataset] = dict()
    for init_Z_method in init_Z_runs[dataset].keys():
        hists[dataset][init_Z_method] = list()
        for M, result in init_Z_task_results[dataset][init_Z_method].items():
            outputs = jug.task.value(result)
            for output in outputs:
                hists[dataset][init_Z_method].append(output[-1])


for dataset in init_Z_runs.keys():
    fig, ax = plt.subplots()
    dataset_plot_settings = dict(
        Naval_noisy=dict(elbo_ylim=(35e3, 45e3)),
        Wilson_energy=dict(elbo_ylim=(700, 1050))
    ).get(dataset, dict(elbo_ylim=None))
    for init_Z_method in init_Z_runs[dataset].keys():
        losses = hists[dataset][init_Z_method]
        max_f_evals = max(map(len, losses))
        elbos = -np.array([elbo + [np.nan]*(max_f_evals - len(elbo)) for elbo in losses])
        best_elbo = np.maximum.accumulate(elbos, axis=1)
        #median_elbo = np.nanmedian(best_elbo, axis=0)
        l, = ax.plot(np.arange(len(best_elbo[0])), best_elbo[0], label=f"{init_Z_method} elbo")
        for i in range(1, len(best_elbo)):
            ax.plot(np.arange(len(best_elbo[i])), best_elbo[i], label='_no_legend_', color=l.get_color())
    ax.axhline(baseline_lmls[dataset], label='full GP', linestyle="--")
    ax.legend()
    ax.set_xlabel("Number of Fn. Evals")
    ax.set_ylabel("ELBO")
    ax.set_ylim(dataset_plot_settings['elbo_ylim'])
    ax.set_xlim((-10, 250))
    ax.set_title(dataset)
    fig.savefig(f"./figures/optall-{dataset}-trace.png")
    plt.show()
