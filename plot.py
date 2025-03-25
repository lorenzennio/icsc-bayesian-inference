import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import ticker

def plot_2d_pdf_contour(
        pdf_function,
        x_range=(0.3, 1.7),
        y_range=(1.9, 2.1),
        resolution=500,
        x_sigma=None,
        y_sigma=None,
        x_label=r'$\theta_0$',
        y_label=r'$\theta_1$',
        mode_label=r'$(\hat\theta_0, \hat \theta_1)$',
        title=None,
        file=None
        ):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.vectorize(lambda x, y: pdf_function(x, y))(X, Y)

    levels = np.min(Z) + np.array([1,4,9,16])
    plt.contour(X, Y, Z, cmap='Purples_r', levels=levels)
    ind_Zmin = np.unravel_index(np.argmin(Z), Z.shape)
    plt.plot(X[ind_Zmin], Y[ind_Zmin], 'x', label=mode_label)
    if x_sigma:
        plt.vlines([x_sigma[0]-x_sigma[1], x_sigma[0]+x_sigma[1]], *y_range, alpha=0.5, label=r'$\pm 1\sigma$')
        plt.xticks([x_sigma[0]+ i * x_sigma[1] for i in range(-3,4,1)], rotation=45)
    if y_sigma:
        plt.hlines([y_sigma[0]-y_sigma[1], y_sigma[0]+y_sigma[1]], *x_range, alpha=0.5)
        plt.yticks([y_sigma[0]+ i * y_sigma[1] for i in range(-3,4,1)], rotation=45)

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    c = plt.colorbar(label=r'$N^2\sigma$')
    c.set_ticklabels([1,4,9,16])
    plt.legend()
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file, bbox_inches='tight')
    plt.show()