# inducing-init

## Code guidelines
- Tests can be run using ```pytest -x --cov-report html --cov=inducing_init```.
- Some scripts are paralellised using `jug`.
  - Make sure you install it using `pip install jug`.
  - You can run all the tasks in a script in parallel using `jug execute jug_script.py`.
  - Jug communicates over the filesystem, so multiple computers can paralellise the same script if they share a networked filesystem.
  - Usually, a separate script takes care of the plotting / processing of the results.

## Todo
- Will probably have to rename this repo at some point to denote its true scope: Robust optimisation of GP models.
