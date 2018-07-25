from scipy.optimize import curve_fit
import sep
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from fitFunc import *
import csv
import itertools
import json
import os
import shutil

configList = json.load(open("configList.json"))
######parameter list######
cropRange = configList["findSpotParameters"]["cropRange"]
# Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C
guessUpBound = configList["fittingParameters"]["guessUpBound"]
guessLowBound = configList["fittingParameters"]["guessLowBound"]
if configList["fittingParameters"]["smartGuessUpBound"]:
    guessUpBound[1] = (cropRange+1)*2
    guessUpBound[2] = (cropRange+1)*2
guessBound = (guessLowBound, guessUpBound)
#    sigma_x,sigma_y,theta,A,B,C
intConfigGuess = configList["fittingParameters"]["intGuess"]
######parameter list######

def makeResultDir():
    if not os.path.exists(os.path.join(os.curdir, "Result")):
        os.makedirs(os.path.join(os.curdir, "Result"))
        print("make Result Dir")

def makeShiftCenterResultDir(dataFolderName):
    if not os.path.exists(os.path.join(dataFolderName, "ShiftCenterResult")):
        os.makedirs(os.path.join(dataFolderName, "ShiftCenterResult"))
        print("make ShiftCenterResult Dir")


def copyJsontoLog(timeStamp):
    if not os.path.exists(os.path.join(os.curdir, "Log")):
        os.makedirs(os.path.join(os.curdir, "Log"))
        print("make Log Dir")

    sourceDirectory = os.curdir
    newFileName = timeStamp + "_"+configList["csvNameRemark"]+ ".json"
    finalDirectory = os.path.join(os.curdir, "Log")
    dstFile = os.path.join(finalDirectory, newFileName)
    sourceFile = os.path.join(sourceDirectory, "configList.json")
    shutil.copy(sourceFile, dstFile)
    print("Copied Json file to Log")


def plotFunc(plot_data, plotSensitivity=3):
    m, s = np.mean(plot_data), np.std(plot_data)
    plt.imshow(plot_data, interpolation='nearest', cmap='jet',
               vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
               origin='lower')
    plt.colorbar()
    plt.show()


def setPicDim(filePath):
    global picWidth, picHeight
    data = np.array(Image.open(filePath))
    picWidth = len(data[1])
    picHeight = len(data)
    print("Width: ", picWidth, ", Height: ", picHeight)
    print("Image Center: ", picWidth / 2, picHeight / 2)


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


# def plotSpots(imgArray, objects_list, plotSensitivity=3, saveMode=False, saveFileName="test", showSpots=False):
#     # plot background-subtracted image
#     fig, ax = plt.subplots()
#     m, s = np.mean(imgArray), np.std(imgArray)
#     plt.imshow(imgArray, interpolation='nearest', cmap='gray',
#                vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
#                origin='lower')
#
#     # plot an ellipse for each object
#     for i in range(len(objects_list)):
#         e = Ellipse(xy=(objects_list['x'][i], objects_list['y'][i]),
#                     width=6 * objects_list['a'][i],
#                     height=6 * objects_list['b'][i],
#                     angle=objects_list['theta'][i] * 180. / np.pi)
#         e.set_facecolor('none')
#         e.set_edgecolor('red')
#         ax.add_artist(e)
#
#     plt.colorbar()
#     if saveMode:
#         savePath = configList["saveFigModeParameters"]["saveFigFolderName"]
#         plt.savefig(savePath + saveFileName + ".png")
#
#     if showSpots:
#         plt.show()
#     else:
#         plt.clf()

# Tony: Change the plot anatomy
def plotSpots(imgArray, objects_list, plotSensitivity_low=0.0, plotSensitivity_up=0.5,
              saveMode=False, saveFileName="test", showSpots=False):
    fig, ax = plt.subplots()
    min_int, max_int = np.amin(imgArray), np.amax(imgArray)
    plt.imshow(imgArray, interpolation='nearest', cmap='jet',
               vmin=min_int + (max_int - min_int) * plotSensitivity_low,
               vmax=min_int + (max_int - min_int) * plotSensitivity_up,
               origin='lower')

    # plot an ellipse for each object
    for i in range(len(objects_list)):
        e = Ellipse(xy=(objects_list['x'][i], objects_list['y'][i]),
                    width=3 * objects_list['a'][i],
                    height=3 * objects_list['b'][i],
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


# def getSpotRoughRange(imgArray: np.array, searchThreshold: float, mask: np.array,
#                       scaleDownFactor: float = 10, plotSensitivity: float = 3, showSpots: bool = False,
#                       fullInformation: bool = False, saveMode=False, saveFileName="test") -> np.array:
@jit
def getSpotRoughRange(imgArray: np.array, searchThreshold: float, mask: np.array, scaleDownFactor: float = 10,
                      plotSensitivity_low: float = 0.0, plotSensitivity_up: float = 0.5,
                      showSpots: bool = False, fittingMode: bool = False, saveMode=False,
                      saveFileName="test") -> np.array:
    imgArray = compressImage(imgArray, scaleDownFactor)
    imgArray = applyMask(imgArray, mask)

    bkg = sep.Background(imgArray)
    objects_list = sep.extract(imgArray, searchThreshold, err=bkg.globalrms)

    if showSpots is True or saveMode is True:
        plotSpots(imgArray, objects_list, plotSensitivity_low, plotSensitivity_up,
                  showSpots=showSpots, saveMode=saveMode, saveFileName=saveFileName)

    if fittingMode is True:
        return np.array([objects_list['x'], objects_list['y']]).T

    else:
        return np.array([objects_list['peak'], objects_list['x'], objects_list['y'],
                         objects_list['xmax'], objects_list['ymax'],
                         objects_list['a'], objects_list['b'], objects_list['theta']]).T


def plotFitFunc(fit_params):

    xi, yi = np.mgrid[:cropRange * 2:30j, :cropRange * 2:30j]
    xyi = np.vstack([xi.ravel(), yi.ravel()])

    zpred = fitFunc(xyi, *fit_params)
    zpred.shape = xi.shape

    fig, ax = plt.subplots()

    im = ax.imshow(zpred, extent=(xi.min(), xi.max(), yi.max(), yi.min()), aspect='auto')
    fig.colorbar(im)
    ax.invert_yaxis()
    plt.show()


def fitCurve(imageArray, centerArray, plotFittedFunc=False, printParameters=False):
    global cropRange, guessBound, intConfigGuess
    allFittedSpot = []

    for i in range(len(centerArray)):
        spotNumber = i
        xyzArray = []

        cropedArray = imageArray[
                      int(centerArray[spotNumber][1]) - cropRange: int(centerArray[spotNumber][1]) + cropRange,
                      int(centerArray[spotNumber][0]) - cropRange: int(centerArray[spotNumber][0]) + cropRange]

        for i in range(len(cropedArray)):
            for j in range(len(cropedArray[i])):
                xyzArray.append([i, j, cropedArray[i][j]])

        x, y, z = np.array(xyzArray).T
        xy = x, y
        i = z.argmax()

        intGuess = [z[i], x[i], y[i]]
        intGuess = intGuess + intConfigGuess
        pred_params, uncert_cov = curve_fit(fitFunc, xy, z, p0=intGuess, bounds=guessBound)


        ####do cord transform
        pred_params[1] = pred_params[1] - cropRange + centerArray[spotNumber][0]
        pred_params[2] = pred_params[2] - cropRange + centerArray[spotNumber][1]

        pred_params = pred_params.tolist()
        allFittedSpot.append(pred_params)

        if plotFittedFunc == True: plotFitFunc(pred_params)
        if printParameters == True: print(pred_params)
        # Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C


    return allFittedSpot


def saveToCSV(RowArray, fileName):
    with open(fileName, 'a', newline='') as f:
        csvWriter = csv.writer(f)
        for i in RowArray:
            csvWriter.writerow(i)



@jit
def findSpot(fileName, searchThreshold, mask, showSpots=False, plotSensitivity_low=0.0, plotSensitivity_up=0.5,
             scaleDownFactor=1,
             plotFittedFunc=False, printParameters=False, fileID=0, saveMode=False, fittingMode=True, shiftCenterMode = False):
    imageArray = readLEEDImage(fileName)
    returnArray = []
    centerArray = getSpotRoughRange(imageArray, searchThreshold, mask, scaleDownFactor=scaleDownFactor,
                                    showSpots=showSpots,
                                    plotSensitivity_low=plotSensitivity_low, plotSensitivity_up=plotSensitivity_up,
                                    saveMode=saveMode, saveFileName=fileName, fittingMode=fittingMode)
    if fittingMode:
        returnArray.append(
            fitCurve(imageArray, centerArray, plotFittedFunc=plotFittedFunc, printParameters=printParameters))
        returnList = list(itertools.chain.from_iterable(returnArray))
        returnList = list(itertools.chain.from_iterable(returnList))
        elements = int(len(returnList) / 9)
    else:
        returnArray.append(centerArray)
        returnList = list(itertools.chain.from_iterable(returnArray))
        returnList = list(itertools.chain.from_iterable(returnList))
        elements = int(len(returnList) / 8)

    returnList.insert(0, elements)
    returnList.insert(0, fileName)
    returnList.insert(0, fileID)
    if shiftCenterMode:
        return returnList, elements, imageArray
    else:
        return returnList, elements

def saveImArrayTo(imageArray,fullPathAndFileName):
    saveArray = Image.fromarray(imageArray)
    saveArray.save(fullPathAndFileName)