import jug.task
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

jug.init("jug_search_uci.py", "jug_search_uci.jugdata")
from jug_search_uci import (
    dataset_names, sparse_task_results, get_settings, baseline_results, baseline_exps, sparse_exps
)

plot_all_datasets = False
plot_normalised = True

# Can comment this out to run all datasets
if not plot_all_datasets:
    # dataset_names = ["Wilson_energy", "Wilson_concrete", "Wilson_airfoil", "Wilson_wine"]
    # dataset_names = [n for n in dataset_names if n not in
    #                  ["Wilson_pendulum", "Pendulum_noisy", "Wilson_wine"]]
    dataset_names = [n for n in dataset_names if n not in
                     ["Wilson_pendulum", "Pendulum_noisy", "Wilson_wine", "kin40k"]]
    # dataset_names = ["Naval", "Naval_noisy"]
    # dataset_names = ["Wilson_stock", "Wilson_energy", "Wilson_concrete", "Wilson_airfoil"]

# Get values from tasks
sparse_results_raw = {}
sparse_results_normalised = {}
baseline_lmls = {}
for dataset_name in dataset_names:
    if (type(baseline_results[dataset_name]) is float) or not baseline_results[dataset_name].can_load():
        continue
    baseline_lmls[dataset_name] = jug.task.value(baseline_results[dataset_name])

    experiment_storage_path, Ms, common_run_settings, dataset_custom_settings = get_settings(dataset_name)
    sparse_task_values = [jug.task.value(result) for result in sparse_task_results[dataset_name]]
    sparse_results_raw[dataset_name] = pd.DataFrame.from_records(
        sparse_task_values, columns=['elbo', 'upper', 'rmse', 'nlpp'], index=Ms
    )
    noise_model_lml = len(baseline_exps[dataset_name].X_train) * (-0.5 * np.log(2 * np.pi) - 0.5)
    sparse_results_normalised[dataset_name] = sparse_results_raw[dataset_name].copy()
    sparse_results_normalised[dataset_name].elbo -= baseline_lmls[dataset_name]
    sparse_results_normalised[dataset_name].elbo /= baseline_lmls[dataset_name] - noise_model_lml
    sparse_results_normalised[dataset_name].upper -= baseline_lmls[dataset_name]
    sparse_results_normalised[dataset_name].index /= baseline_exps[dataset_name].X_train.shape[0]

    baseline_exps[dataset_name].load()
    print(f"{dataset_name:30} lik variance: {baseline_exps[dataset_name].model.likelihood.variance.numpy():.8f}"
          f" lml: {baseline_lmls[dataset_name]}")

sparse_results = sparse_results_normalised if plot_normalised else sparse_results_raw

_, ax = plt.subplots()
for dataset_name in sparse_results.keys():
    # ax.axhline(baseline_lmls[dataset_name])
    l, = ax.plot(sparse_results[dataset_name].index, sparse_results[dataset_name].elbo,
                 label=f"{dataset_name} ({len(sparse_exps[dataset_name][0].X_train)})")
    # ax.plot(sparse_results[dataset_name].index, sparse_results[dataset_name].upper,
    #         color=l.get_color(), linestyle=':')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='x-small', ncol=5)
plt.show()
