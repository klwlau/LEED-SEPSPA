import numpy as np


def fitFunc(xy,Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C):
#     global cccounter
#     cccounter+=1
    x,y=xy
    theta=np.deg2rad(theta)
    a=np.cos(theta)**2/(2*sigma_x**2)+np.sin(theta)**2/(2*sigma_y**2)
    b=-np.sin(2*theta)/(4*sigma_x**2)+np.sin(2*theta)/(4*sigma_y**2)
    c=np.sin(theta)**2/(2*sigma_x**2)+np.cos(theta)**2/(2*sigma_y**2)
    g= Amp*np.exp(-(a*(x-x_0)**2+2*b*(x-x_0)*(y-y_0)+c*(y-y_0)**2))
    g+= A*x + B*y + C

    return g
