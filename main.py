from matplotlib import rcParams
from UsedFunc import *
from fitFunc import *
rcParams['figure.figsize'] = [10., 8.]


#read Image and make Mask
fileArray = readLEEDImage("test2.tif")# to set the picWidth,picHeight for findSpot function
mask = makeMask(125, 125, 0, 30)
centerArray = findSpot(fileArray, 100, mask, scaleFactor=10,showSpots=True, plotSensitivity=4)


fitCurve(fileArray,centerArray)


print("done")