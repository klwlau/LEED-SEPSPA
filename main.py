from matplotlib import rcParams
from UsedFunc import *
import glob
import datetime
import time
rcParams['figure.figsize'] = [10., 8.]


def mainLoop():
    writeBufferArray = []
    counter = 0;
    fileAmount = len(fileList)
    for fileName in fileList:
        counter += 1
        templist, elements = findSpot(fileName, 60, mask, scaleFactor=1, showSpots=False)
        writeBufferArray.append(templist)
        print(counter, ",", elements, ",", fileName, ",", counter / fileAmount * 100, "%")
        if counter % writeBuffer == 0:
            saveToCSV(writeBufferArray, CSVName)
            writeBufferArray = []
        if counter == fileAmount:
            saveToCSV(writeBufferArray, CSVName)


# int parameter, make Mask
timeStamp=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
CSVName=timeStamp+".csv"
folderName="20180212_scan01"
fileList = glob.glob("./"+folderName+"/*.tif")
setPicDim(fileList[0]) # to set the picWidth,picHeight for findSpot function
mask = makeMask(470, 440, 250, 300) # int mask
writeBuffer =20


# findSpot(fileList[0], 130, mask, scaleFactor=1,showSpots=True,plotSensitivity=4)



mainLoop()
print("done")
