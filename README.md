# RobustGP
Procedures for robust initialisation and optimisation of Variational Sparse Gaussian processes. This code accompanies 
Burt et al (2019, 2020) (see sitation below), and implements the recommendations.

## The bottom line
In Burt et al (2020), we recommend Sparse GP Regression (SGPR) (Titsias, 2009) models to be trained in the following way:
- Initialise the inducing inputs using the ```ConditionalVariance``` methods.
- Alternately optimise the hyperparameters only and reinitialise the inducing inputs using ```ConditionalVariance```.
  See ```FullbatchUciExperiment``` for an example on how to implement this (```training_procedure == 'reinit_Z'```).

We find that when using ```ConditionalVariance``` we obtain the same performance as gradient-optimised inducing inputs
with a slightly larger number of inducing varianbles. The benefit is not having to do gradient-based optimisation, which
is often more of a pain than it is worth. 

A few anecdotal suggestions for practitioners:
- We suggest using ```ConditionalVariance``` even for non-Gaussian likelihoods for initialisation, although you may want
  test yourself whether to use the periodic reinitialisation method, or gradient-based inducing input optimisation.
- When getting Cholesky errors, consider reinitialising the inducing inputs with ```ConditionalVariance``` rather than
  e.g. raising jitter. ```ConditionalVariance``` will repel the inducing inputs based on any new hyperparameters which
  caused high correlation between old inducing variables, leading to better conditioning of ```Kuu```.
  
### Example
```
M = 1000  # We choose 1000 inducing variables
k = gpflow.kernels.SquaredExponential()
# Initialise hyperparameters here
init_method = robustgp.ConditionalVariance()
Z = init_method.compute_initialisation(X, M, k)[0]
model = gpflow.models.SGPR((X_train, Y_train), k, Z)
for _ in range(10):
    # Optimise w.r.t. hyperparmeters here...
    Z = init_method.compute_initialisation(X, M, k)[0]  # Reinit with the new kernel hyperparameters
    self.model.inducing_variable.Z = gpflow.Parameter(Z)
```
  
## What the code provides
### Inducing input initialisation
We provide various inducing point initialisation methods, together with some tools for robustly optimising GPflow 
models. We really only recommend using ```ConditionalVariance``` for initialising inducing inputs, with the others being
included for the experiments in the paper.

### Automatic jitter selection
In addition, we provide versions of the GPflow classes ```SGPR``` and ```GPR``` that have objective functions that are
robust to Cholesky/inversion errors. This is implemented by automatic increasing of jitter, as is done in e.g. 
[GPy](https://sheffieldml.github.io/GPy/). This process is a bit cumbersome in TensorFlow, and to do it we provide the
classes ```RobustSGPR``` and ```RobustGPR```, as well as a customised Scipy optimiser ```RobustScipy```. To see how this
is used, see the class ```FullbatchUciExperiment``` in the ```robustgp_experiments``` directory. 

### Experiments
All the experiments from Burt et al (2020) are included  in the ```robustgp_experiments``` directory.

## Code guidelines
For using the initialisation code:
- Make sure that [GPflow](https://github.com/GPflow/GPflow) is installed, followed by running ```pip setup.py develop```.
- Tests can be run using ```pytest -x --cov-report html --cov=robustgp```.

For running the experiments
- We use code from [Bayesian benchmarks](https://github.com/hughsalimbeni/bayesian_benchmarks) to handle dataset
  loading. Some assembly needed to get all the datasets.
- Some scripts are paralellised using `jug`.
  - Make sure it's installed using `pip install jug`.
  - You can run all the tasks in a script in parallel by running `jug execute jug_script.py` multiple times.
  - Jug communicates over the filesystem, so multiple computers can paralellise the same script if they share a networked filesystem.
  - Usually, a separate script takes care of the plotting / processing of the results.

## Citation
To cite the recommendations in our paper or this accompanying software, please refer to our JMLR paper.
```
@article{burt2020gpviconv,
  author  = {David R. Burt and Carl Edward Rasmussen and Mark van der Wilk},
  title   = {Convergence of Sparse Variational Inference in Gaussian Processes Regression},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {131},
  pages   = {1-63},
  url     = {http://jmlr.org/papers/v21/19-1015.html}
}
```

This JMLR paper is an extended version of our ICML paper.
```
@InProceedings{burt2019gpviconv,
  title = 	 {Rates of Convergence for Sparse Variational {G}aussian Process Regression},
  author = 	 {Burt, David and Rasmussen, Carl Edward and van der Wilk, Mark},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {862--871},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/burt19a/burt19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/burt19a.html},
}
```
