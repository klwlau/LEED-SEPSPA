from matplotlib import rcParams
from UsedFunc import *
import glob
rcParams['figure.figsize'] = [10., 8.]



# int parameter, make Mask
CSVName="spot.csv"
fileList = glob.glob("./*.tif")
setPicDim("test2.tif") # to set the picWidth,picHeight for findSpot function
mask = makeMask(125, 125, 0, 30)
writeBuffer =20

# need to rewrite mainloop
writeBufferArray=[]
counter=0;fileAmount= len(fileList)
for fileName in fileList:
    counter += 1
    templist = createRowArray(fileName, mask)
    writeBufferArray.append(templist)
    print(counter/fileAmount*100,"%")
    if counter % writeBuffer == 0:
        saveToCSV(writeBufferArray, CSVName)
        writeBufferArray = []
    if counter == fileAmount:
        saveToCSV(writeBufferArray, CSVName)

print("done")
