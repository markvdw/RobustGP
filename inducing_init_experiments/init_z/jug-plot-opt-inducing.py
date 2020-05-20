import jug.task
import matplotlib.pyplot as plt
import numpy as np

jug.init("jug_opt_inducing.py", "jug_opt_inducing.jugdata")
from jug_opt_inducing import (init_Z_runs, init_Z_task_results, baseline_lmls)

hists = {}
for dataset in init_Z_runs.keys():
    hists[dataset] = dict()
    for init_Z_method in init_Z_runs[dataset].keys():
        hists[dataset][init_Z_method] = dict()
        for M, result in init_Z_task_results[dataset][init_Z_method].items():
            _, _, _, _, hist = jug.task.value(result)[0]
            hists[dataset][init_Z_method] = hist


for dataset in init_Z_runs.keys():
    fig, ax = plt.subplots()
    for init_Z_method in init_Z_runs[dataset].keys():
        ax.plot(hists[dataset][init_Z_method][0], -np.array(hists[dataset][init_Z_method][1]), label=f"{init_Z_method} elbo")
    ax.axhline(baseline_lmls[dataset], label='full GP', linestyle="--")
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    ax.set_title(dataset)
    fig.savefig(f"./figures/optall-{dataset}-trace.png")
