from fitFunc import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pytictoc import TicToc

timer = TicToc()

xSpace = np.linspace(-10, 10, 100)
ySpace = np.linspace(-10, 10, 100)

xi, yi = np.meshgrid(xSpace, ySpace)
xyi = np.vstack([xi.ravel(), yi.ravel()])

guessUpBound = [15, 15, 5000, 100000, 10, 10, 30, 30, 180, 100000, 10, 10, 30, 30, 180]
guessLowBound = [-15, -15, 0, 0, -10, -10, 0.001, 0.001, -180, 0, -10, -10, 0.001, 0.001, -180]

guessBound = [guessLowBound, guessUpBound]



def plottingOriFunc():
    zSpace.shape = xi.shape
    plt.imshow(zSpace)
    plt.colorbar()
    plt.show()


def fittingTest():
    # for i in range(1):
    numberOfGauss = 2
    fittingParamsSampleArray = np.ones(numberOfGauss * 6 + 3)
    fit_params, uncert_cov = curve_fit(NGauss(numberOfGauss), xyi, zSpace,
                                       tuple(fittingParamsSampleArray), bounds=guessBound)

for i in range(100):
    zSpace = NGauss(2)(xyi, 0, 2, 0, 300, 0, 5, 4, 2, 120, 300, 3, 0, 1, 2, 120)
    zSpace += + np.random.rand(len(zSpace))
    timer.tic()
    fittingTest()
    timer.toc()

