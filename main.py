from matplotlib import rcParams
from UsedFunc import *
import glob
import datetime
import time
rcParams['figure.figsize'] = [10., 8.]



# int parameter, make Mask
timeStamp=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
CSVName=timeStamp+".csv"
folderName="20180212_scan01"
fileList = glob.glob("./"+folderName+"/*.tif")
setPicDim(fileList[0]) # to set the picWidth,picHeight for findSpot function
mask = makeMask(450, 450, 0, 100) # int mask
writeBuffer =20

# need to rewrite mainloop
writeBufferArray=[]
counter=0;fileAmount= len(fileList)
for fileName in fileList:
    counter += 1
    templist = findSpot(fileName, 100, mask, scaleFactor=1,showSpots=False)
    writeBufferArray.append(templist)
    print(counter,",",fileName,",",counter/fileAmount*100,"%")
    if counter % writeBuffer == 0:
        saveToCSV(writeBufferArray, CSVName)
        writeBufferArray = []
    if counter == fileAmount:
        saveToCSV(writeBufferArray, CSVName)

print("done")
