import numpy as np
from numba import jit
from scipy.stats import norm

@jit
def pdf_skewnormal(x, location=0.0, scale=1.0, shape=0.0):
    scale = scale ** 2
    t = (x - location) / scale
    return 2.0 / scale * norm.pdf(t) * norm.cdf(shape * t)


def backGroundPlaneEstimation(x, y, A, B, C):
    return A * x + B * y + C


def gauss2D(x, y, Amp, x_0, y_0, sigma_x, sigma_y, theta):
    theta = np.deg2rad(theta)
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)
    g = Amp * np.exp(-(a * (x - x_0) ** 2 + 2 * b * (x - x_0) * (y - y_0) + c * (y - y_0) ** 2))
    return g


@jit
def fitFunc(xy, Amp, x_0, y_0, sigma_x, sigma_y, theta, A, B, C):
    x, y = xy
    g = gauss2D(x, y, Amp, x_0, y_0, sigma_x, sigma_y, theta)
    g += backGroundPlaneEstimation(x, y, A, B, C)
    return g


def NGauss(numOfGauss):
    def makeNGauss(xy, *parameters):
        xi, yi = xy
        g = 0
        gaussParams = parameters[3:]
        g += backGroundPlaneEstimation(xi, yi, parameters[0], parameters[1], parameters[2])
        for i in range(numOfGauss):
            g += gauss2D(xi, yi, gaussParams[i * 6], gaussParams[i * 6 + 1], gaussParams[i * 6 + 2],
                         gaussParams[i * 6 + 3], gaussParams[i * 6 + 4], gaussParams[i * 6 + 5])
        return g

    return makeNGauss

# @jit
# def fit2GaussFunc(xy, Amp_1, x_0_1, y_0_1, sigma_x_1, sigma_y_1, theta_1, Amp_2, x_0_2, y_0_2, sigma_x_2, sigma_y_2,
#                   theta_2, A, B, C):
#     x, y = xy
#     g = gauss2D(x, y, Amp_1, x_0_1, y_0_1, sigma_x_1, sigma_y_1, theta_1)
#     g += gauss2D(x, y, Amp_2, x_0_2, y_0_2, sigma_x_2, sigma_y_2, theta_2)
#     g += backGroundPlaneEstimation(x, y, A, B, C)
#     return g
