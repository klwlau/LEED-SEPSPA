import time
import glob

start_time = time.time()
print("Program Started, Loading Libraries")
import datetime

print("Start Loading UsedFunc")
from UsedFunc import *


def printSaveStatus():
    global globalCounter
    if globalCounter != 0:
        elapsedTime = ((time.time() - start_time) / 60)
        totalTime = elapsedTime / (globalCounter / totalFileNumber)
        timeLeft = totalTime - elapsedTime

        print("---Elapsed Time: %.2f / %.2f Minutes ---" % (elapsedTime, totalTime)
              + "---Time Left: %.2f  Minutes ---" % timeLeft
              + "--save to" + CSVName)


def printCurrentStatus(counter, numberOfSpots, filePath):
    print(counter, ", ", "%.2f" % (counter / totalFileNumber * 100), "%, ", numberOfSpots, ",", filePath)


def fittingMode():
    global globalCounter
    # init first row in CSV file
    # writeBufferArray for 2D normal distribution
    writeBufferArray = [["FileID", "File Name", "Number of Spots", "Elapsed Time",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "theta", "A", "B", "C", "Chi^2"]]

    #
    globalCounter = 0
    fileAmount = len(fileList)

    for filePath in fileList:
        timer.tic()
        # need to add all parameters back
        templist, numberOfSpots = findSpot(filePath, configList["findSpotParameters"]["searchThreshold"],
                                           mask, scaleDownFactor=configList["findSpotParameters"]["scaleDownFactor"],
                                           showSpots=False, fileID=globalCounter,
                                           saveFitFuncPlot=configList["fittingParameters"]["saveFitFuncPlot"])
        templist.insert(3, timer.tocvalue())
        writeBufferArray.append(templist)

        # print(counter, ",", numberOfSpots, ",", filePath, ",", counter / fileAmount * 100, "%")
        printCurrentStatus(globalCounter, numberOfSpots, filePath)

        if globalCounter % CSVwriteBuffer == 0:
            saveToCSV(writeBufferArray, CSVName)
            writeBufferArray = []
            printSaveStatus()

        if globalCounter == (fileAmount - 1):
            saveToCSV(writeBufferArray, CSVName)
            printSaveStatus()
        globalCounter += 1


def sepMode():
    global globalCounter
    # init first row in CSV file
    writeBufferArray = [["FileID", "File Name", "Number of Spots",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta"]]
    globalCounter = 0
    fileAmount = len(fileList)

    for filePath in fileList:
        # need to add all parameters back
        templist, numberOfSpots = findSpot(filePath, configList["findSpotParameters"]["searchThreshold"],
                                           mask, scaleDownFactor=configList["findSpotParameters"]["scaleDownFactor"],
                                           showSpots=False, fileID=globalCounter,
                                           fittingMode=False)
        writeBufferArray.append(templist)

        # print(counter, ",", numberOfSpots, ",", fileName, ",", counter / fileAmount * 100, "%")
        printCurrentStatus(globalCounter, numberOfSpots, filePath)

        if globalCounter % CSVwriteBuffer == 0:
            saveToCSV(writeBufferArray, CSVName)
            writeBufferArray = []
            printSaveStatus()

        if globalCounter == (fileAmount - 1):
            saveToCSV(writeBufferArray, CSVName)
            printSaveStatus()
        globalCounter += 1


def testMode():
    # need to add testMode parameters
    print("File name: ", fileList[configList["testModeParameters"]["testModeFileID"]])
    findSpot(fileList[configList["testModeParameters"]["testModeFileID"]],
             configList["findSpotParameters"]["searchThreshold"], mask,
             scaleDownFactor=configList["testModeParameters"]["scaleDownFactor"],
             plotSensitivity_low=configList["testModeParameters"]["plotSensitivity_low"],
             plotSensitivity_up=configList["testModeParameters"]["plotSensitivity_up"],
             showSpots=configList["testModeParameters"]["showSpots"],
             plotFittedFunc=configList["testModeParameters"]["plotFittedFunc"],
             printFittedParameters=configList["testModeParameters"]["printFittedParameters"],
             printSpotRoughRangeArray=configList["testModeParameters"]["printSpotRoughRangeArray"],
             fittingMode=configList["testModeParameters"]["fittingMode"])


print("---Initializing---")
# setup
timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
dataFolderName = configList["dataFolderName"]
makeResultDir()
CSVName = "./Result/" + timeStamp + "_" + configList["csvNameRemark"] + ".csv"
copyJsontoLog(timeStamp)

# int parameter, make Mask, read file name in folderimport json
if not dataFolderName:
    fileList = glob.glob("./*.tif")
else:
    fileList = glob.glob(dataFolderName + "/*.tif")
fileList = sorted(fileList)

# fileList = fileList[:30]

globalCounter = 0
totalFileNumber = len(fileList)

setPicDim(fileList[0])  # to set the picWidth,picHeight for findSpot function
mask = makeMask(configList["maskConfig"]["mask_x_center"], configList["maskConfig"]["mask_y_center"],
                configList["maskConfig"]["innerRadius"], configList["maskConfig"]["outerRadius"])  # int mask
CSVwriteBuffer = configList["CSVwriteBuffer"]

if configList["testMode"]:
    print("testMode")
    testMode()
else:
    if configList["fittingMode"]:
        print("fittingMode")
        fittingMode()
        print("save to :" + CSVName)
    else:
        print("sepMode")
        sepMode()
        print("save to :" + CSVName)

print("--- Total Time: %.2f Minutes ---" % ((time.time() - start_time) / 60))
print("done")

# errorList = np.array(errorList)
# plt.plot(errorList)
# plt.show()
# plt.hist(errorList)
# plt.show()
