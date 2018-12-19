from scipy.optimize import curve_fit
import sep
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pytictoc import TicToc
import csv, itertools, json, os, shutil, ntpath
from fitFunc import *

timer = TicToc()

configList = json.load(open("configList.json"))
######parameter list######
cropRange = configList["findSpotParameters"]["cropRange"]
# Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C
guessUpBound = configList["fittingParameters"]["guessUpBound"]
guessLowBound = configList["fittingParameters"]["guessLowBound"]

guessBound = [guessLowBound, guessUpBound]
dataFolderName = configList["dataFolderName"]
#    sigma_x,sigma_y,theta,A,B,C
intConfigGuess = configList["fittingParameters"]["intGuess"]
errorList = []


######parameter list######

def makeResultDir():
    if not os.path.exists(os.path.join(os.curdir, "Result")):
        os.makedirs(os.path.join(os.curdir, "Result"))
        print("make Result Dir")


def makeShiftCenterResultDir(dataFolderName):
    if not os.path.exists(os.path.join(dataFolderName, "ShiftCenterResult")):
        os.makedirs(os.path.join(dataFolderName, "ShiftCenterResult"))
        print("make ShiftCenterResult Dir")


def makeDirInDataFolder(dataFolderName, dirName):
    if not os.path.exists(os.path.join(dataFolderName, dirName)):
        os.makedirs(os.path.join(dataFolderName, dirName))
        print("make ", dirName, " Dir")
    return os.path.join(dataFolderName, dirName)


def copyJsontoLog(timeStamp):
    if not os.path.exists(os.path.join(os.curdir, "Log")):
        os.makedirs(os.path.join(os.curdir, "Log"))
        print("make Log Dir")

    sourceDirectory = os.curdir
    newFileName = timeStamp + "_" + configList["csvNameRemark"] + ".json"
    finalDirectory = os.path.join(os.curdir, "Log")
    dstFile = os.path.join(finalDirectory, newFileName)
    sourceFile = os.path.join(sourceDirectory, "configList.json")
    shutil.copy(sourceFile, dstFile)
    print("Copied Json file to Log")


def plotArray(plot_data, plotSensitivity=3):
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


def genFittedFuncArray(fit_params, cropRange, outputZpredOnly=False, ):
    xi, yi = np.mgrid[0:cropRange * 2, 0:cropRange * 2]

    xyi = np.vstack([xi.ravel(), yi.ravel()])

    zpred = fitFunc(xyi, *fit_params)

    zpred.shape = xi.shape

    if outputZpredOnly:
        return zpred
    else:
        return xi, yi, zpred


@jit
def calRSquareError(fittedArray, rawArray):
    error = fittedArray - rawArray
    errorSquare = error ** 2
    numberOfElement = fittedArray.size
    return np.sum(errorSquare) / numberOfElement


@jit
def calChiSquareError(fittedArray, rawArray):
    error = fittedArray - rawArray
    errorSquare = error ** 2
    # numberOfElement = fittedArray.size
    return np.sum(errorSquare / rawArray)


def calMeanError(zpred, cropArray, meanArea=10):
    # zpred = np.array(zpred)
    # tempArray = zpred[2]
    center = int(zpred[0].shape[0] / 2)
    zpred = zpred[center - meanArea:center + meanArea,
            center - meanArea:center + meanArea]
    cropArray = cropArray[center - meanArea:center + meanArea,
                center - meanArea:center + meanArea]
    # plotArray(zpred - cropArray)

    return ((zpred - cropArray) ** 2).mean()


def plotFitFunc(fit_params, cropedArray, plotSensitivity=5, saveFitFuncPlot=False, saveFitFuncFileName="fitFuncFig"):
    global dataFolderName, configList

    Chi_square = fit_params[-1]
    fit_params = fit_params[:-1]

    xi, yi, zpred = genFittedFuncArray(fit_params, cropRange)

    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    m, s = np.mean(cropedArray), np.std(cropedArray)
    cs = ax1.imshow(cropedArray, interpolation='nearest', cmap='jet',
                    vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
                    origin='lower')
    fig.colorbar(cs)
    plt.title("Chi^2= %.2f" % (Chi_square))
    ax1.contour(yi, xi, zpred,
                vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s, alpha=1, origin='lower')  # cmap='jet',
    if saveFitFuncPlot:
        if saveFitFuncFileName == "fitFuncFig":
            plt.savefig(saveFitFuncFileName + ".png")
        else:
            saveFigFullPath = makeDirInDataFolder(dataFolderName, "fitFuncFig_"
                                                  + configList["fittingParameters"]["saveFitFuncPlotFileRemark"])
            plt.savefig(saveFigFullPath + "/" + saveFitFuncFileName + ".png")
        plt.close(fig)
        return

    plt.show()


@jit
def getSpotRoughRange(imgArray: np.array, searchThreshold: float, mask: np.array, scaleDownFactor: float = 10,
                      plotSensitivity_low: float = 0.0, plotSensitivity_up: float = 0.5,
                      showSpots: bool = False, fittingMode: bool = False, saveMode=False, printReturnArray=False,
                      saveFileName="test") -> np.array:
    imgArray = compressImage(imgArray, scaleDownFactor)
    imgArray = applyMask(imgArray, mask)

    bkg = sep.Background(imgArray)
    objects_list = sep.extract(imgArray, searchThreshold, err=bkg.globalrms)

    if showSpots is True or saveMode is True:
        plotSpots(imgArray, objects_list, plotSensitivity_low, plotSensitivity_up,
                  showSpots=showSpots, saveMode=saveMode, saveFileName=saveFileName)

    if fittingMode is True:
        returnArray = np.array([objects_list['xcpeak'], objects_list['ycpeak']]).T
        # returnArray = np.array([objects_list['x'], objects_list['y']]).T

        if printReturnArray:
            print(len(returnArray))
            print(returnArray)
        return returnArray, objects_list

    else:
        returnArray = np.array([objects_list['peak'], objects_list['x'], objects_list['y'],
                                objects_list['xmax'], objects_list['ymax'],
                                objects_list['a'], objects_list['b'], objects_list['theta']]).T
        if printReturnArray:
            print(len(returnArray))
            print(returnArray)
        return returnArray


def fitCurve(imageArray, centerArray, objectList, plotFittedFunc=False, printFittedParameters=False,
             saveFitFuncPlot=False, saveFitFuncFileName=""):
    global cropRange, guessBound, intConfigGuess, configList, errorList
    allFittedSpot = []

    for spotNumber in range(len(centerArray)):
        RSquare = 100000
        adcropRange = cropRange

        while RSquare > configList["fittingParameters"]["ChiSqThreshold"]:
            xyzArray = []
            cropedArray = imageArray[
                          int(centerArray[spotNumber][1]) - adcropRange: int(centerArray[spotNumber][1]) + adcropRange,
                          int(centerArray[spotNumber][0]) - adcropRange: int(centerArray[spotNumber][0]) + adcropRange]

            for xx in range(len(cropedArray)):
                for yy in range(len(cropedArray[xx])):
                    xyzArray.append([xx, yy, cropedArray[xx][yy]])

            x, y, z = np.array(xyzArray).T
            xy = x, y
            i = z.argmax()

            intGuess = [z[i], x[i], y[i]]
            intConfigGuess[0] = objectList[spotNumber]["a"]
            intConfigGuess[1] = objectList[spotNumber]["b"]
            intConfigGuess[2] = np.rad2deg(objectList[spotNumber]["theta"])
            intGuess = intGuess + intConfigGuess

            if configList["fittingParameters"]["smartXYGuessBound"]:
                guessBound[0][1] = x[i] - configList["fittingParameters"]["smartXYGuessBoundRange"]
                guessBound[1][1] = x[i] + configList["fittingParameters"]["smartXYGuessBoundRange"]
                guessBound[0][2] = y[i] - configList["fittingParameters"]["smartXYGuessBoundRange"]
                guessBound[1][2] = y[i] + configList["fittingParameters"]["smartXYGuessBoundRange"]

            fit_params, uncert_cov = curve_fit(fitFunc, xy, z, p0=intGuess, bounds=guessBound)
            RSquare = calChiSquareError(genFittedFuncArray(fit_params, adcropRange, outputZpredOnly=True), cropedArray)
            adcropRange -= 2

        fit_params = fit_params.tolist()
        fit_params.append(RSquare)
        # print(RSquare)

        if plotFittedFunc: plotFitFunc(fit_params, cropedArray)
        if saveFitFuncPlot == True: plotFitFunc(fit_params, cropedArray, saveFitFuncPlot=saveFitFuncPlot,
                                                saveFitFuncFileName=saveFitFuncFileName + "_" + str(spotNumber))

        ####do cord transform
        fit_params[1] = fit_params[1] - adcropRange + centerArray[spotNumber][0]
        fit_params[2] = fit_params[2] - adcropRange + centerArray[spotNumber][1]

        allFittedSpot.append(fit_params)

        if printFittedParameters == True: print("Fitted Parameters :", fit_params)
        # Amp,x_0,y_0,sigma_x,sigma_y,theta,A,B,C,R^2

    return allFittedSpot


def saveToCSV(RowArray, fileName):
    with open(fileName, 'a', newline='') as f:
        csvWriter = csv.writer(f)
        for i in RowArray:
            csvWriter.writerow(i)


@jit
def findSpot(filePath, searchThreshold, mask, showSpots=False, plotSensitivity_low=0.0, plotSensitivity_up=0.5,
             scaleDownFactor=1,
             plotFittedFunc=False, printFittedParameters=False, fileID=0, fittingMode=True,
             shiftCenterMode=False, printSpotRoughRangeArray=False, saveFitFuncPlot=False):
    imageArray = readLEEDImage(filePath)
    fileName = ntpath.basename(filePath)[:-4]
    returnArray = []
    centerArray, objectList = getSpotRoughRange(imageArray, searchThreshold, mask, scaleDownFactor=scaleDownFactor,
                                                showSpots=showSpots,
                                                plotSensitivity_low=plotSensitivity_low,
                                                plotSensitivity_up=plotSensitivity_up,
                                                saveFileName=filePath, fittingMode=fittingMode,
                                                printReturnArray=printSpotRoughRangeArray)

    if fittingMode:
        returnArray.append(
            fitCurve(imageArray, centerArray, objectList, plotFittedFunc=plotFittedFunc,
                     printFittedParameters=printFittedParameters,
                     saveFitFuncPlot=saveFitFuncPlot, saveFitFuncFileName=fileName))
        returnList = list(itertools.chain.from_iterable(returnArray))
        returnList = list(itertools.chain.from_iterable(returnList))
        numberOfSpots = int(len(returnList) / 10)
    else:
        returnArray.append(centerArray)
        returnList = list(itertools.chain.from_iterable(returnArray))
        returnList = list(itertools.chain.from_iterable(returnList))
        numberOfSpots = int(len(returnList) / 8)

    returnList.insert(0, numberOfSpots)
    returnList.insert(0, filePath)
    returnList.insert(0, fileID)
    if shiftCenterMode:
        return returnList, numberOfSpots, imageArray
    else:
        return returnList, numberOfSpots


def saveImArrayTo(imageArray, fullPathAndFileName):
    saveArray = Image.fromarray(imageArray)
    saveArray.save(fullPathAndFileName)
