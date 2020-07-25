import jug.task
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
font = {"family": "cmr10", "size": 24}
matplotlib.rc("font", **font)
matplotlib.rc("text", usetex=True)

color_dict = {"Kmeans": "#ff7f00", "Greedy Conditional Variance": "C9", "Gradient": "#377eb8"}
name_dict = {
    "Kmeans": "K-means (reinit.)",
    "Greedy Conditional Variance": "Greedy var. (reinit.)",
    "Gradient": "Gradient",
}
plot_title_dict = {
    "Wilson_energy": "Energy (M=65)",
    "Wilson_elevators": "Elevators (M=1200)",
    "Naval_noisy": "Naval with Noise (M=55)",
}
jug.init("jug_opt_inducing.py", "jug_opt_inducing.jugdata")
from jug_opt_inducing import init_Z_runs, init_Z_task_results, baseline_lmls

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
        Naval_noisy=dict(elbo_ylim=(32e3, 39e3)),
        Wilson_energy=dict(elbo_ylim=(700, 1050)),
        Wilson_elevators=dict(elbo_ylim=(-7000, -6000)),
    ).get(dataset, dict(elbo_ylim=None))
    for init_Z_method in init_Z_runs[dataset].keys():
        losses = hists[dataset][init_Z_method]
        max_f_evals = max(map(len, losses))
        elbos = -np.array([elbo + [np.nan] * (max_f_evals - len(elbo)) for elbo in losses])
        best_elbo = np.maximum.accumulate(elbos, axis=1)
        # median_elbo = np.nanmedian(best_elbo, axis=0)
        (l,) = ax.plot(
            np.arange(len(best_elbo[0])), best_elbo[0], label=name_dict[init_Z_method], color=color_dict[init_Z_method]
        )
        for i in range(1, len(best_elbo)):
            ax.plot(np.arange(len(best_elbo[i])), best_elbo[i], label="_no_legend_", color=l.get_color())
    ax.axhline(baseline_lmls[dataset], label="Full GP", linestyle="--", color="k")
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # ax.legend()
    ax.set_xlabel("Number of Fn. Evals.")
    ax.set_ylabel("ELBO")
    ax.set_ylim(dataset_plot_settings["elbo_ylim"])
    ax.set_xlim((-10, 250))
    ax.set_title(plot_title_dict[dataset])
    plt.tight_layout()
    fig.savefig(f"./figures/optall-{dataset}-trace.pdf")
    plt.show()
