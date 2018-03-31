print("Program Started, Loading Libraries")
from matplotlib import rcParams
from UsedFunc import *
import glob
import datetime
import time

# rcParams['figure.figsize'] = [10., 8.]


def mainLoop():
    #init first row in CSV file
    writeBufferArray = [["FileID","File Name","Number of Spots","Am","x_0","y_0","sigma_x","sigma_y","theta","A","B","C","Am","x_0","y_0","sigma_x","sigma_y","theta","A","B","C","Am","x_0","y_0","sigma_x","sigma_y","theta","A","B","C","Am","x_0","y_0","sigma_x","sigma_y","theta","A","B","C","Am","x_0","y_0","sigma_x","sigma_y","theta","A","B","C","Am","x_0","y_0","sigma_x","sigma_y","theta","A","B","C","Am","x_0","y_0","sigma_x","sigma_y","theta","A","B","C"]]
    counter = 0
    fileAmount = len(fileList)
    for fileName in fileList:

        #need to add all parameters back
        templist, numberOfSpots = findSpot(fileName, configList["findSpotParameters"]["searchThreshold"]
                                           , mask, scaleDownFactor=configList["findSpotParameters"]["scaleDownFactor"],
                                           showSpots=False, fileID=counter)
        writeBufferArray.append(templist)

        print(counter, ",", numberOfSpots, ",", fileName, ",", counter / fileAmount * 100, "%")
        if counter % CSVwriteBuffer == 0:
            saveToCSV(writeBufferArray, CSVName)
            writeBufferArray = []
        if counter == fileAmount:
            saveToCSV(writeBufferArray, CSVName)
        counter += 1

print("---Initializing---")
start_time = time.time()


#setup
timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M')
CSVName = timeStamp + ".csv"
folderName= configList["folderName"]
# int parameter, make Mask, read file name in folder
if not folderName:
    fileList=glob.glob("./*.tif")
else:
    fileList = glob.glob("./" + folderName + "/*.tif")

setPicDim(fileList[0])  # to set the picWidth,picHeight for findSpot function
mask = makeMask(configList["maskConfig"]["mask_x_center"], configList["maskConfig"]["mask_y_center"]
                , configList["maskConfig"]["innerRadius"], configList["maskConfig"]["outerRadius"])  # int mask
CSVwriteBuffer = configList["CSVwriteBuffer"]


if configList["testMode"]:
    print("testMode")
    #need to add testMode parameters
    print("File name: ",fileList[configList["testModeParameters"]["testModeFileID"]])
    findSpot(fileList[configList["testModeParameters"]["testModeFileID"]], configList["findSpotParameters"]["searchThreshold"],
             mask,
             scaleDownFactor = configList["testModeParameters"]["scaleDownFactor"], plotSensitivity=3,
             showSpots=configList["testModeParameters"]["showSpots"],
             plotFittedFunc= configList["testModeParameters"]["plotFittedFunc"],
             printParameters=configList["testModeParameters"]["printParameters"])
else:
    print("fittingMode")
    mainLoop()


print("--- %s Minutes ---" % ((time.time() - start_time)/60))
print("done")
