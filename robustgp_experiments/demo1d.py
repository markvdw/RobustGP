import gpflow
import matplotlib.pyplot as plt
import numpy as np
from robustgp import ConditionalVariance

X = np.random.rand(150, 1)
Y = 0.8 * np.cos(10 * X) + 1.2 * np.sin(8 * X + 0.3) + np.cos(17 * X) * 1.2 + np.random.randn(*X.shape) * 0.1

gpr = gpflow.models.GPR((X, Y), gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(gpr.training_loss, gpr.trainable_variables, options=dict(maxiter=100))

k = gpflow.kernels.SquaredExponential()
gpflow.utilities.multiple_assign(k, gpflow.utilities.read_values(gpr.kernel))

Z_initer = ConditionalVariance()
sp = gpflow.models.SGPR((X, Y), k, Z_initer.compute_initialisation(X, 6, k)[0])
gpflow.utilities.multiple_assign(sp, gpflow.utilities.read_values(gpr))

pX = np.linspace(0, 1, 3000)[:, None]
m, v = sp.predict_f(pX)
ipm, _ = sp.predict_f(sp.inducing_variable.Z.value())

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(X, Y, 'x')
ax1.plot(pX, m)
ax1.plot(sp.inducing_variable.Z.value(), ipm, 'o', color='C3')
deviation = (2 * (v + sp.likelihood.variance.value()) ** 0.5).numpy().flatten()
ax1.fill_between(pX.flatten(), m.numpy().flatten() - deviation, m.numpy().flatten() + deviation, alpha=0.3)
ax1.axvline(pX[np.argmax(v)].item(), color='C2')
ax1.set_ylabel("y")
ax2.plot(pX, v ** 0.5)
ax2.plot(sp.inducing_variable.Z.value(), sp.inducing_variable.Z.value() * 0.0, 'o', color='C3')
ax2.axvline(pX[np.argmax(v)].item(), color='C2')
ax2.set_xlabel("input $x$")
ax2.set_ylabel("$\mathbb{V}\,[p(f(x) | \mathbf{u}]^{0.5}$")
plt.show()
