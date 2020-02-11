from inducing_experiments.utils import ExperimentRecord

import gpflow

gpflow.config.set_default_positive_minimum(1e-5)

MAXITER = 6000

experiment_name = "regression"
dataset_names = ["Wilson_elevators",
                 "Naval",
                 "Wilson_tamielectric",
                 "WineWhite",
                 "kin40k"]

for dataset_name in dataset_names:
    experiment_storage_path = f"./storage-{experiment_name}/{dataset_name}"

    linear_run_setting = {"model_class": "SGPR", "M": 100, "kernel_class": "Linear", "fixed_Z": True}
    ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name, **linear_run_setting)

    basic_run_settings_list = [
        # {"model_class": "GPR"},
        {"model_class": "SGPR", "M": 100, "fixed_Z": True},
        {"model_class": "SGPR", "M": 200, "fixed_Z": True},
        {"model_class": "SGPR", "M": 500, "fixed_Z": True},
        {"model_class": "SGPR", "M": 1000, "fixed_Z": True},
        # {"model_class": "SGPR", "M": 2000, "fixed_Z": True},
        # {"model_class": "SGPR", "M": 5000, "fixed_Z": True},
    ]

    common_params = {"storage_path": experiment_storage_path, "dataset_name": dataset_name}
    baseline_runs = [ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name, **basic_run_settings)
                     for basic_run_settings in basic_run_settings_list]
    [r.cached_run(MAXITER) for r in baseline_runs]

    # uniform_init_settings_list = [
    #     {"model_class": "SGPR", "M": 100, "fixed_Z": True, "init_Z_method": "uniform", "restart_id": i} for i in range(10)
    # ]

    greedy_init_settings_list = [
        {"model_class": "SGPR", "M": 100, "fixed_Z": True, "init_Z_method": "greedy-trace"},
        {"model_class": "SGPR", "M": 200, "fixed_Z": True, "init_Z_method": "greedy-trace"},
        {"model_class": "SGPR", "M": 500, "fixed_Z": True, "init_Z_method": "greedy-trace"},
        {"model_class": "SGPR", "M": 1000, "fixed_Z": True, "init_Z_method": "greedy-trace"},
        # {"model_class": "SGPR", "M": 2000, "fixed_Z": True, "init_Z_method": "greedy-trace"},
        # {"model_class": "SGPR", "M": 5000, "fixed_Z": True, "init_Z_method": "greedy-trace"},
    ]
    greedy_init_runs = [ExperimentRecord(storage_path=experiment_storage_path, dataset_name=dataset_name,
                                         **run_settings)
                        for run_settings in greedy_init_settings_list]
    [r.cached_run(MAXITER) for r in greedy_init_runs]
    break


