def print_post_run(run):
    print("")
    try:
        std_ratio = (run.model.kernel.variance.numpy() / run.model.likelihood.variance.numpy()) ** 0.5
        print(f"(kernel.variance / likelihood.variance)**0.5: {std_ratio}")
        print(run.model.kernel.lengthscales.numpy())
        print(f"ELBO: {run.model.elbo().numpy()}")
    except AttributeError:
        pass
    print("")
    print("")


uci_train_settings = dict(
    Wilson_bike=([100, 200, 500, 1000, 2000, 5000], {}),
    Wilson_elevators=([100, 200, 500, 1000, 2000, 5000], {}),
    Wilson_pol=([100, 200, 500, 1000, 2000, 5000], {}),
    Power=([100, 200, 500, 1000, 2000, 5000], {}),  # Step function in it?
    # Kin8mn=([100, 200, 500, 1000, 2000], {}),  # Can't download
    Parkinsons_noisy=([100, 150, 170, 200, 500], {}),
    Wilson_parkinsons=([100, 150, 170, 200, 500], {}),
    Wilson_sml=([100, 200, 500, 1000, 2000, 3000, 3500], {}),  # Mostly linear, but with benefit of nonlinear
    # Didn't get SE+Lin working, probably local optimum
    # Wilson_skillcraft=([10, 20, 50, 100, 200, 500], {"kernel_name": "SquaredExponentialLinear"}),
    Wilson_skillcraft=([10, 20, 50, 100, 200, 500, 1000], {}),  # Mostly linear, but with benefit of nonlinear
    Wilson_gas=([100, 200, 500, 1000, 1300], {}),
    Naval=([10, 20, 50, 100, 200], {}),  # Very sparse solution exists
    Naval_noisy=([10, 20, 50, 100, 200, 500], {}),  # Very sparse solution exists
    Wilson_wine=([100, 200, 500, 1000, 1300], {}),  # Suddenly catches good hypers with large M
    Wilson_airfoil=([100, 200, 500, 1000, 1250, 1300, 1340], {}),  # Good
    Wilson_solar=([100, 200, 300],
                  {"kernel_name": "SquaredExponentialLinear", "max_lengthscale": 10.0}),  # Mostly linear
    # Good, better performance with Linear kernel added
    # Wilson_concrete=([100, 200, 500, 600, 700, 800, 900],
    #                  {"kernel_name": "SquaredExponentialLinear", "optimizer": "bfgs", "max_lengthscale": 10.0}),
    Wilson_concrete=([100, 200, 500, 600, 700, 800, 900], {}),
    Wilson_pendulum=([10, 100, 200, 500, 567], {}),  # Not sparse, due to very low noise
    Pendulum_noisy=([10, 100, 200, 500, 567], {}),  # Not sparse, due to very low noise
    Wilson_forest=([10, 100, 200, 400], {"kernel_name": "SquaredExponentialLinear"}),  # Bad
    Wilson_energy=([10, 50, 100, 200, 500], {}),  # Good
    Wilson_stock=([10, 50, 100, 200, 400, 450], {"kernel_name": "SquaredExponentialLinear"}),  # Mostly linear
    Wilson_housing=([100, 200, 300, 400], {})  # Bad
)
