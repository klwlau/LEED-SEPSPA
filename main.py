from matplotlib import rcParams
from UsedFunc import *
from fitFunc import *
rcParams['figure.figsize'] = [10., 8.]


#read Image and make Mask
fileArray = readLEEDImage("test2.tif")# to set the picWidth,picHeight for findSpot function
mask = makeMask(125, 125, 0, 30)
centerArray = findSpot(fileArray, 100, mask, scaleFactor=10,showSpots=True, plotSensitivity=4)




mask = makeMask(125, 125, 0, 30)

fitCurve(fileArray,centerArray)
# for i in range(len(centerArray)):
#     spotNumber = i
#     # print(centerArray[spotNumber])
#     cropRange = 8
#     spot1 = fileArray[int(centerArray[spotNumber][1]) - cropRange:int(centerArray[spotNumber][1]) + cropRange,\
#             int(centerArray[spotNumber][0]) - cropRange:int(centerArray[spotNumber][0]) + cropRange]
#         # plotFunc(spot1)
#
#
#     cropedArray = spot1
#     xyzArray = []
#     for i in range(len(cropedArray)):
#         for j in range(len(cropedArray[i])):
#             xyzArray.append([i, j, cropedArray[i][j]])
#
#     x, y, z = np.array(xyzArray).T
#     xy = x, y
#     i = z.argmax()
#     guess = [z[i], x[i], y[i], 50, 50, 100, 30, 30, 100]
#     pred_params, uncert_cov = curve_fit(fitFunc, xy, z, p0=guess, method='lm')
#     # plotFitFunc(xy, 11, pred_params)
#     print(pred_params)

print("done")