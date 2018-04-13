import time

start_time = time.time()
print("Program Started, Loading Libraries")
from UsedFunc import *
from modeFunc import *
import glob
import datetime
import time




print("---Initializing---")
# setup
timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
dataFolderName = configList["dataFolderName"]
CSVName = timeStamp + ".csv"

# int parameter, make Mask, read file name in folder
if not dataFolderName:
    fileList = glob.glob("./*.tif")
else:
    fileList = glob.glob(dataFolderName + "/*.tif")

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
    else:
        print("sepMode")
        sepMode()


print("--- %s Minutes ---" % ((time.time() - start_time) / 60))
print("done")
print("save to :" + CSVName)
