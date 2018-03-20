from matplotlib import rcParams
from UsedFunc import *
import glob
import datetime
import time
rcParams['figure.figsize'] = [10., 8.]



# int parameter, make Mask
timeStamp=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
CSVName=timeStamp+".csv"
folderName="20180213_scan02"
fileList = glob.glob("./"+folderName+"/*.tif")
setPicDim(fileList[0]) # to set the picWidth,picHeight for findSpot function
mask = makeMask(125, 125, 0, 30) # int mask
writeBuffer =20

# need to rewrite mainloop
writeBufferArray=[]
counter=0;fileAmount= len(fileList)
for fileName in fileList:
    counter += 1
    templist = findSpot(fileName, 18, mask, scaleFactor=10)
    writeBufferArray.append(templist)
    print(counter,",",fileName,",",counter/fileAmount*100,"%")
    if counter % writeBuffer == 0:
        saveToCSV(writeBufferArray, CSVName)
        writeBufferArray = []
    if counter == fileAmount:
        saveToCSV(writeBufferArray, CSVName)

print("done")
