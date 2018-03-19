from matplotlib import rcParams
from UsedFunc import *
rcParams['figure.figsize'] = [10., 8.]


#int parameter, make Mask
fileArray = readLEEDImage("test2.tif")# to set the picWidth,picHeight for findSpot function
mask = makeMask(125, 125, 0, 30)


#potential mainloop
fileArray = readLEEDImage("test2.tif")
centerArray = findSpot(fileArray, 100, mask, scaleFactor=10,showSpots=False, plotSensitivity=4)

fitCurve(fileArray,centerArray)


print("done")