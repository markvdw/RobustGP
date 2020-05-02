import jug.task
import matplotlib.pyplot as plt
import pandas as pd

jug.init("jug_search_uci.py", "jug_search_uci.jugdata")
from jug_search_uci import (
    dataset_names, sparse_task_results, get_settings, baseline_results, baseline_exps
)

plot_all_datasets = False
plot_normalised = False

# Can comment this out to run all datasets
if not plot_all_datasets:
    # dataset_names = ["Wilson_energy", "Wilson_concrete", "Wilson_airfoil", "Wilson_wine"]
    dataset_names = [n for n in dataset_names if n not in ["Wilson_pendulum", "Pendulum_noisy"]]

# Get values from tasks
sparse_results_raw = {}
sparse_results_normalised = {}
baseline_lmls = {}
for dataset_name in dataset_names:
    baseline_lmls[dataset_name] = jug.task.value(baseline_results[dataset_name])

    experiment_storage_path, Ms, common_run_settings, dataset_custom_settings = get_settings(dataset_name)
    sparse_task_values = [jug.task.value(result) for result in sparse_task_results[dataset_name]]
    sparse_results_raw[dataset_name] = pd.DataFrame.from_records(
        sparse_task_values, columns=['elbo', 'upper', 'rmse', 'nlpp'], index=Ms
    )
    sparse_results_normalised[dataset_name] = sparse_results_raw[dataset_name].copy()
    sparse_results_normalised[dataset_name].elbo -= baseline_lmls[dataset_name]
    sparse_results_normalised[dataset_name].upper -= baseline_lmls[dataset_name]
    sparse_results_normalised[dataset_name].index /= baseline_exps[dataset_name].X_train.shape[0]

sparse_results = sparse_results_normalised if plot_normalised else sparse_results_raw

_, ax = plt.subplots()
for dataset_name in dataset_names:
    # ax.axhline(baseline_lmls[dataset_name])
    l, = ax.plot(sparse_results[dataset_name].index, sparse_results[dataset_name].elbo,
                 label=dataset_name)
    ax.plot(sparse_results[dataset_name].index, sparse_results[dataset_name].upper,
            color=l.get_color(), linestyle=':')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='x-small', ncol=5)
plt.show()
# #
# #
# # Plotting
# greedy_rmses = [np.mean((exp.model.predict_f(exp.X_test)[0].numpy() - exp.Y_test) ** 2.0) ** 0.5
#                 for exp in greedy_init_runs]
# greedy_nlpps = [-np.mean(exp.model.predict_log_density((exp.X_test, exp.Y_test)))
#                 for exp in greedy_init_runs]
#
# if run_gpr:
#     full_rmse = np.mean((gpr_exp.model.predict_f(gpr_exp.X_test)[0].numpy() - gpr_exp.Y_test) ** 2.0) ** 0.5
#     full_nlpp = -np.mean(gpr_exp.model.predict_log_density((gpr_exp.X_test, gpr_exp.Y_test)))
# else:
#     full_rmse, full_nlpp = np.nan, np.nan
# print(f"gpr rmse: {full_rmse}")
# print(f"rmse    : {greedy_rmses}")
# print(f"gpr nlpp: {full_nlpp}")
# print(f"nlpp    : {greedy_nlpps}")
#
# m_elbo, m_rmse, m_nlpp = baselines.meanpred_baseline(None, greedy_init_runs[0].Y_train, None,
#                                                      greedy_init_runs[0].Y_test)
# l_elbo, l_rmse, l_nlpp = baselines.linear_baseline(greedy_init_runs[0].X_train, greedy_init_runs[0].Y_train,
#                                                    greedy_init_runs[0].X_test, greedy_init_runs[0].Y_test)

# _, ax = plt.subplots()
# # plt.plot([r.M for r in baseline_runs], [r.model.elbo().numpy() for r in baseline_runs], '-x')
# ax.plot([r.M for r in greedy_init_runs], [r.model.elbo().numpy() for r in greedy_init_runs], '-x')
# ax.plot([r.M for r in greedy_init_runs], [r.model.upper_bound().numpy() for r in greedy_init_runs], '-x')
# if run_gpr:
#     ax.axhline(gpr_exp.model.log_marginal_likelihood().numpy(), linestyle="--")
# ax.set_xlabel("M")
# ax.set_ylabel("elbo")
#
# _, ax = plt.subplots()
# # [plt.plot(*r.train_objective_hist) for r in baseline_runs]
# [ax.plot(*r.train_objective_hist) for r in greedy_init_runs]
# if run_gpr:
#     ax.axhline(-gpr_exp.model.log_marginal_likelihood().numpy(), linestyle="--")
# ax.axhline(-m_elbo, linestyle=':')
# ax.axhline(-l_elbo, linestyle='-.')
# ax.set_xlabel("iters")
# ax.set_ylabel("elbo")
#
# _, ax = plt.subplots()
# ax.plot(Ms, greedy_rmses)
# ax.axhline(m_rmse, linestyle=':')
# ax.axhline(l_rmse, linestyle='-.')
# ax.axhline(full_rmse, linestyle='--')
# ax.set_xlabel("M")
# ax.set_ylabel("rmse")
#
# _, ax = plt.subplots()
# ax.plot(Ms, greedy_nlpps)
# ax.axhline(m_nlpp, linestyle=':')
# ax.axhline(l_nlpp, linestyle='-.')
# ax.axhline(full_nlpp, linestyle='--')
# ax.set_xlabel("M")
# ax.set_ylabel("nlpp")
#
# plt.show()
