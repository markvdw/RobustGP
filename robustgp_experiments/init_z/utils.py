import tensorflow as tf


def print_post_run(run):
    print("")
    try:
        print(f"ELBO: {run.model.robust_maximum_log_likelihood_objective().numpy()}")
        std_ratio = (run.model.kernel.variance.numpy() / run.model.likelihood.variance.numpy()) ** 0.5
        print(f"(kernel.variance / likelihood.variance)**0.5: {std_ratio}")
        print(run.model.kernel.lengthscales.numpy())
    except AttributeError:
        pass
    except tf.errors.InvalidArgumentError as e:
        print("")
        print("Probably a CholeskyError:")
        print(e.message)
        print("Ignoring...")
    print("")
    print("")


uci_train_settings = dict(
    Wilson_tamielectric=([100, 200, 500, 1000, 2000], {}),
    Wilson_protein=([100, 200, 500, 1000, 2000], {}),
    Wilson_kin40k=([(1000, 2000, 5000, 10000, 15000), {}]),
    Wilson_bike=([100, 200, 500, 1000, 2000, 5000], {}),
    Wilson_elevators=([100, 200, 500, 1000, 2000, 5000], {}),
    Wilson_pol=([100, 200, 500, 1000, 2000, 5000], {}),
    Power=([100, 200, 500, 1000, 2000, 5000], {}),  # Step function in it?
    # Kin8mn=([100, 200, 500, 1000, 2000], {}),  # Can't download
    Parkinsons_noisy=([100, 150, 170, 200, 500], {}),
    Wilson_parkinsons=([100, 150, 170, 200, 500, 1000], {}),
    Wilson_sml=([100, 200, 500, 1000, 2000, 3000, 3500], {}),  # Mostly linear, but with benefit of nonlinear
    # Didn't get SE+Lin working, probably local optimum
    # Wilson_skillcraft=([10, 20, 50, 100, 200, 500], {"kernel_name": "SquaredExponentialLinear"}),
    Wilson_skillcraft=([10, 20, 50, 100, 200, 500, 1000], {}),  # Mostly linear, but with benefit of nonlinear
    Wilson_gas=([100, 200, 500, 1000, 1300, 1500], {}),
    Naval=([10, 20, 50, 100, 200], {}),  # Very sparse solution exists
    Naval_noisy=([10, 20, 50, 100, 200, 500], {}),  # Very sparse solution exists
    Wilson_wine=([100, 200, 500, 1000, 1300, 1350], {}),  # Suddenly catches good hypers with large M
    Wilson_airfoil=([100, 200, 500, 800, 1000, 1250, 1300, 1340], {}),  # Good
    Wilson_solar=([100, 200, 300],
                  {"kernel_name": "SquaredExponentialLinear", "max_lengthscale": 10.0}),  # Mostly linear
    # Good, better performance with Linear kernel added
    # Wilson_concrete=([100, 200, 500, 600, 700, 800, 900],
    #                  {"kernel_name": "SquaredExponentialLinear", "optimizer": "bfgs", "max_lengthscale": 10.0}),
    Wilson_concrete=([100, 200, 500, 600, 700, 800, 900], {}),
    Wilson_pendulum=([10, 100, 200, 500, 567], {}),  # Not sparse, due to very low noise
    Pendulum_noisy=([10, 100, 200, 500, 567], {}),  # Not sparse, due to very low noise
    Wilson_forest=([10, 100, 200, 400], {"kernel_name": "SquaredExponentialLinear"}),  # Bad
    Wilson_energy=([10, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500], {}),  # Good
    Wilson_stock=([10, 50, 100, 200, 400, 450], {"kernel_name": "SquaredExponentialLinear"}),  # Mostly linear
    Wilson_housing=([100, 200, 300, 400], {}),  # Bad
    Wilson_yacht=([10, 20, 50, 100, 200, 250], {}),
    Wilson_autompg=([10, 20, 50, 100, 200, 250], {}),
    Wilson_servo=([10, 20, 30, 40, 50, 70, 100, 110, 120, 130, 140], {}),
    Wilson_breastcancer=([10, 50, 100, 150], {}),
    Wilson_autos=([10, 20, 50, 100], {}),
    Wilson_concreteslump=([10, 20, 50, 60, 70], {})
)

bad_datasets = ["Wilson_housing", "Wilson_forest"]
good_datasets = [k for k in uci_train_settings.keys() if k not in bad_datasets]
