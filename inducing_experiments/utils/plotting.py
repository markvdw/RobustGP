import matplotlib.pyplot as plt
import numpy as np


def plot_1d_model(m, *, data=None):
    D = m.inducing_variable.Z.numpy().shape[1]
    if data is not None:
        X, Y = data[0], data[1]
        plt.plot(X, Y, 'x')

    data_inducingpts = np.vstack((X if data else np.zeros((0, D)), m.inducing_variable.Z.numpy()))
    pX = np.linspace(np.min(data_inducingpts) - 1.0, np.max(data_inducingpts) + 1.0, 300)[:, None]
    pY, pYv = m.predict_y(pX)

    line, = plt.plot(pX, pY, lw=1.5)
    col = line.get_color()
    plt.plot(pX, pY + 2 * pYv ** 0.5, col, lw=1.5)
    plt.plot(pX, pY - 2 * pYv ** 0.5, col, lw=1.5)
    plt.plot(m.inducing_variable.Z.numpy(), np.zeros(m.inducing_variable.Z.numpy().shape), 'k|', mew=2)
