from matplotlib import rcParams
from UsedFunc import *
import glob
rcParams['figure.figsize'] = [10., 8.]



    
    

# int parameter, make Mask
fileList = glob.glob("./*.tif")
setPicDim("test2.tif") # to set the picWidth,picHeight for findSpot function
mask = makeMask(125, 125, 0, 30)

# need to rewrite mainloop
print(createSaveArray("test2.tif", mask))


print("done")
