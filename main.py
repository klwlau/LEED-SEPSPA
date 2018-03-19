import numpy as np
from scipy.optimize import curve_fit
import sep
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import glob
import csv
rcParams['figure.figsize'] = [10., 8.]
from UsedFunc import *
from fitFunc import *

testfileA = readLEEDImage("test2.tif")
mask = makeMask(125, 125, 0, 30)
centerArray = findSpot("test2.tif", 1000, mask, showSpots=False, plotSensitivity=4)

testfileA = readLEEDImage("test2.tif")
mask = makeMask(125, 125, 0, 30)
centerArray = findSpot("test2.tif", 100, mask, showSpots=True, plotSensitivity=4)

for i in range(len(centerArray)):
    spotNumber = i
    print(centerArray[spotNumber])
    cropRange = 8
    spot1 = testfileA[int(centerArray[spotNumber][1]) - cropRange:int(centerArray[spotNumber][1]) + cropRange, \
            int(centerArray[spotNumber][0]) - cropRange:int(centerArray[spotNumber][0]) + cropRange]
    #     plotFunc(spot1)


    testArray = spot1
    xyzArray = []
    for i in range(len(testArray)):
        for j in range(len(testArray[i])):
            xyzArray.append([i, j, testArray[i][j]])

    x, y, z = np.array(xyzArray).T
    xy = x, y
    i = z.argmax()
    guess = [z[i], x[i], y[i], 50, 50, 100, 30, 30, 100]
    pred_params, uncert_cov = curve_fit(fitFunc, xy, z, p0=guess, method='lm')
    plotFitFunc(xy, 11, pred_params)
    #     print(pred_params)