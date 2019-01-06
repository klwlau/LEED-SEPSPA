from fitFunc import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pytictoc import TicToc

timer = TicToc()

xSpace = np.linspace(-10, 10, 100)
ySpace = np.linspace(-10, 10, 100)

xi, yi = np.meshgrid(xSpace, ySpace)
xyi = np.vstack([xi.ravel(), yi.ravel()])

backgroundGuessUpperBound = [15, 15, 5000]
backgroundGuessLowerBound = [-15, -15, 0]

gaussianUpperBoundTemplate = [100000, 10, 10, 30, 30, 180]
gaussianLowerBoundTemplate = [0, -10, -10, 0.001, 0.001, -180]


def plottingOriFunc():
    zSpace.shape = xi.shape
    plt.imshow(zSpace)
    plt.colorbar()
    plt.show()


def genBound(numOfGauss):
    # , backgroundGuessUpBound=backgroundGuessUpBound, backgroundGuessLowBound=backgroundGuessLowBound,
    #              guessUpBoundTemp=guessUpBoundTemp, guessLowBoundTemp=guessLowBoundTemp):


    guessUpBound = backgroundGuessUpperBound.copy()
    guessLowBound = backgroundGuessLowerBound.copy()

    for num in range(numOfGauss):
        guessUpBound += gaussianUpperBoundTemplate
        guessLowBound += gaussianLowerBoundTemplate

    return [guessLowBound, guessUpBound]


def fittingTest():
    # for i in range(1):
    numberOfGauss = 3
    fittingParamsSampleArray = np.ones(numberOfGauss * 6 + 3)
    fit_params, uncert_cov = curve_fit(NGauss(numberOfGauss), xyi, zSpace,
                                       tuple(fittingParamsSampleArray), bounds=genBound(numberOfGauss))
    print(fit_params)

for i in range(3):
    zSpace = NGauss(2)(xyi, 0, 2, 0, 300, 0, 5, 4, 2, 120, 300, 3, 0, 1, 2, 120)
    zSpace += + np.random.rand(len(zSpace))
    timer.tic()
    fittingTest()
    timer.toc()
