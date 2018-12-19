import numpy as np
from numba import jit
from scipy.stats import norm


@jit
def pdf_skewnormal(x, location=0.0, scale=1.0, shape=0.0):
    scale = scale ** 2
    t = (x - location) / scale
    return 2.0 / scale * norm.pdf(t) * norm.cdf(shape * t)


def backGroundEstimatedPlane(x, y, A, B, C):
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
    g += backGroundEstimatedPlane(x, y, A, B, C)
    return g

@jit
def fit2Gauss(xy, Amp_1, x_0_1, y_0_1, sigma_x_1, sigma_y_1, theta_1, Amp_2, x_0_2, y_0_2, sigma_x_2, sigma_y_2,
              theta_2, A, B, C):
    x, y = xy

# @jit
# def fitFunc(xy, Amp, x_0, y_0, sigma_x, sigma_y, shape_x, shape_y, theta, A, B, C):
#     x, y = xy
#     theta = np.deg2rad(theta)
#     x_rotated = x * np.cos(theta) - y * np.sin(theta)
#     y_rotated = x * np.sin(theta) + y * np.cos(theta)
#     g = Amp * pdf_skewnormal(x_rotated, x_0, sigma_x, shape_x) * \
#         pdf_skewnormal(y_rotated, y_0, sigma_y, shape_y)
#     g += A * x + B * y + C
#     return g
