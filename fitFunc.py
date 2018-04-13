import numpy as np
from numba import jit
from scipy.stats import norm


@jit
def pdf_skewnormal(x, location=0.0, scale=1.0, shape=0.0):
    scale = scale ** 2
    t = (x - location) / scale
    return 2.0 / scale * norm.pdf(t) * norm.cdf(shape * t)


@jit
def fitFunc(xy, Amp, x_0, y_0, sigma_x, sigma_y, shape_x, shape_y, theta, A, B, C):
    x, y = xy
    theta = np.deg2rad(theta)
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)
    g = Amp * pdf_skewnormal(x_rotated, x_0, sigma_x, shape_x) * \
        pdf_skewnormal(y_rotated, y_0, sigma_y, shape_y)
    g += A * x + B * y + C
    return g

# @jit
# def fitFunc(xy,Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C):
# #     global cccounter
# #     cccounter+=1
#     x,y=xy
#     theta=np.deg2rad(theta)
#     a=np.cos(theta)**2/(2*sigma_x**2)+np.sin(theta)**2/(2*sigma_y**2)
#     b=-np.sin(2*theta)/(4*sigma_x**2)+np.sin(2*theta)/(4*sigma_y**2)
#     c=np.sin(theta)**2/(2*sigma_x**2)+np.cos(theta)**2/(2*sigma_y**2)
#     g= Amp*np.exp(-(a*(x-x_0)**2+2*b*(x-x_0)*(y-y_0)+c*(y-y_0)**2))
#     g+= A*x + B*y + C
#
#     return g
