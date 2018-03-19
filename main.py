from matplotlib import rcParams
from UsedFunc import *
import glob
rcParams['figure.figsize'] = [10., 8.]



# int parameter, make Mask
CSVName="spot.csv"
fileList = glob.glob("./*.tif")
setPicDim("test2.tif") # to set the picWidth,picHeight for findSpot function
mask = makeMask(125, 125, 0, 30)

# need to rewrite mainloop
RowArray=[]
counter=0;fileAmount= len(fileList)
for fileName in fileList:
    counter += 1
    templist=createRowArray(fileName, mask)
    RowArray.append(templist)
    print(templist)
    if counter%10 == 0:
        saveToCSV(RowArray,CSVName)
        RowArray = []
    if counter == fileAmount:
        saveToCSV(RowArray,CSVName)
# createSaveArray("test2.tif")

print("done")
