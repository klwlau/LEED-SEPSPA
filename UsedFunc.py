# import numpy as np
from scipy.optimize import curve_fit
import sep
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from fitFunc import *
import csv
import itertools
import json

configList=json.load(open("configList.json"))
######parameter list######
cropRange = configList["findSpotParameters"]["cropRange"]

# Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C
guessUpBound =configList["fittingParameters"]["guessUpBound"]
guessLowBound =configList["fittingParameters"]["guessLowBound"]
guessBound = (guessLowBound,guessUpBound)
#    sigma_x,sigma_y,theta,A,B,C
intConfigGuess= configList["fittingParameters"]["intGuess"]
######parameter list######


def plotFunc(plot_data, plotSensitivity=3):
    m, s = np.mean(plot_data), np.std(plot_data)
    plt.imshow(plot_data, interpolation='nearest', cmap='gray', \
               vmin=m - plotSensitivity * s, \
               vmax=m + plotSensitivity * s, origin='lower')
    plt.colorbar()
    plt.show()


def setPicDim(filePath):
    global picWidth, picHeight
    data = np.array(Image.open(filePath))
    picWidth = len(data[1])
    picHeight = len(data)
    print("Width: ", picWidth, ", Height: ", picHeight)
    print("Image Center: ",picWidth/2,picHeight/2)


def readLEEDImage(filePath):
    data = np.array(Image.open(filePath))
    data = np.flipud(data)
    return data


def makeMask(mask_x_center, mask_y_center, r1, r2):
    global picWidth, picHeight
    mask = [[0 for x in range(picWidth)] for y in range(picHeight)]

    for y in range(picHeight):
        for x in range(picWidth):
            if (x - mask_x_center) ** 2 + (y - mask_y_center) ** 2 > r1 ** 2 and (x - mask_x_center) ** 2 + (
                    y - mask_y_center) ** 2 < r2 ** 2:
                mask[y][x] = 1
    return np.array(mask).astype(np.uint8)


def compressImage(imageArray, scaleFactor):
    imageArray = imageArray / scaleFactor
    #     imageArray=imageArray.astype(np.uint8)
    imageArray = imageArray
    # print(imageArray.dtype)
    return imageArray


def applyMask(imageArray, mask):
    appliedMask = np.multiply(imageArray, mask)
    return appliedMask


def plotSpots(imgArray, objects_list, plotSensitivity=3,saveMode= False,saveFileName="test",showSpots= False):
    # plot background-subtracted image
    fig, ax = plt.subplots()
    m, s = np.mean(imgArray), np.std(imgArray)
    plt.imshow(imgArray, interpolation='nearest', cmap='gray',
                   vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s, origin='lower')

    # plot an ellipse for each object
    for i in range(len(objects_list)):
        e = Ellipse(xy=(objects_list['x'][i], objects_list['y'][i]),
                    width=6 * objects_list['a'][i],
                    height=6 * objects_list['b'][i],
                    angle=objects_list['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)

    plt.colorbar()
    if saveMode:
        savePath = configList["saveFigModeParameters"]["saveFigFolderName"]
        plt.savefig(savePath + saveFileName + ".png")

    if showSpots:
        plt.show()
    else:
        plt.clf()


def getSpotRoughRange(imgArray: np.array, searchThreshold: float, mask: np.array, \
                      scaleDownFactor: float = 10, plotSensitivity: float = 3, showSpots: bool = False, \
                      fullInformation: bool = False,saveMode=False,saveFileName="test") -> np.array:
    # plotFunc(imgArray)
    imgArray = compressImage(imgArray, scaleDownFactor)
    # plotFunc(imgArray)
    imgArray = applyMask(imgArray, mask)
    # plotFunc(imgArray)

    bkg = sep.Background(imgArray)
    objects_list = sep.extract(imgArray, searchThreshold, err=bkg.globalrms)

    if showSpots == True or saveMode == True:
        plotSpots(imgArray, objects_list, plotSensitivity,showSpots=showSpots,saveMode=saveMode,saveFileName=saveFileName)

    if fullInformation == True:
        return objects_list
    else:
        return np.array([objects_list['x'], objects_list['y']]).T


def plotFitFunc(fit_params):  # (xy, zobs, pred_params):
    # x, y = xy
    xi, yi = np.mgrid[:cropRange * 2:30j, :cropRange * 2:30j]
    xyi = np.vstack([xi.ravel(), yi.ravel()])

    zpred = fitFunc(xyi, *fit_params)
    zpred.shape = xi.shape

    fig, ax = plt.subplots()
    #     ax.scatter(x, y, c=zobs, s=200, vmin=zpred.min(), vmax=zpred.max())
    im = ax.imshow(zpred, extent=(xi.min(), xi.max(), yi.max(), yi.min()), aspect='auto')
    fig.colorbar(im)
    ax.invert_yaxis()
    plt.show()


def fitCurve(imageArray, centerArray, plotFittedFunc=False, printParameters=False):
    global cropRange, guessBound,intConfigGuess
    allFittedSpot = []

    for i in range(len(centerArray)):
        spotNumber = i
        xyzArray = []
        # print(centerArray[spotNumber])

        cropedArray = imageArray[
                      int(centerArray[spotNumber][1]) - cropRange: int(centerArray[spotNumber][1]) + cropRange, \
                      int(centerArray[spotNumber][0]) - cropRange: int(centerArray[spotNumber][0]) + cropRange]

        for i in range(len(cropedArray)):
            for j in range(len(cropedArray[i])):
                xyzArray.append([i, j, cropedArray[i][j]])

        x, y, z = np.array(xyzArray).T
        xy = x, y
        i = z.argmax()
        intGuess = [z[i], x[i], y[i], intConfigGuess[0], intConfigGuess[1], intConfigGuess[2],
                    intConfigGuess[3], intConfigGuess[4], intConfigGuess[5]]


        pred_params, uncert_cov = curve_fit(fitFunc, xy, z, p0 = intGuess, bounds=guessBound) #, method='lm' does not support bounds


        ####do cord transform?
        pred_params[1] = pred_params[1] - cropRange + centerArray[spotNumber][0]
        pred_params[2] = pred_params[2] - cropRange + centerArray[spotNumber][1]

        pred_params = pred_params.tolist()
        allFittedSpot.append(pred_params)

        if plotFittedFunc == True: plotFitFunc(pred_params)
        if printParameters == True: print(pred_params)
        # Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C

    ###need to find the center spot

    return allFittedSpot


def saveToCSV(RowArray, fileName):
    with open(fileName, 'a', newline='') as f:
        csvWriter = csv.writer(f)
        for i in RowArray:
            csvWriter.writerow(i)


def findSpot(fileName, searchThreshold, mask, showSpots=False, plotSensitivity=3, scaleDownFactor=10
             , plotFittedFunc=False, printParameters=False, fileID=0,saveMode=False):
    # global mask
    fileArray = readLEEDImage(fileName)
    # fileArray = np.flipud(fileArray)
    returnArray = []
    centerArray = getSpotRoughRange(fileArray, searchThreshold, mask, scaleDownFactor=scaleDownFactor, showSpots=showSpots,
                                    plotSensitivity=plotSensitivity,saveMode=saveMode,saveFileName=fileName)
    returnArray.append(fitCurve(fileArray, centerArray, plotFittedFunc=plotFittedFunc, printParameters=printParameters))
    returnList = list(itertools.chain.from_iterable(returnArray))
    returnList = list(itertools.chain.from_iterable(returnList))
    elements = int(len(returnList) / 9)
    returnList.insert(0, elements)
    returnList.insert(0, fileName)
    returnList.insert(0, fileID)
    return returnList, elements
