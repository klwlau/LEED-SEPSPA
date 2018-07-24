import time
import glob
start_time = time.time()
import datetime
from UsedFunc import *
from PIL import Image
import matplotlib.animation as animation


dataFolderName = configList["dataFolderName"]
fileList = glob.glob(dataFolderName + "/*.tif")
fileList = sorted(fileList)
searchThreshold = 30

setPicDim(fileList[0])

mask = makeMask(configList["maskConfig"]["mask_x_center"], configList["maskConfig"]["mask_y_center"],
                0,1000)

for fileName in fileList:
    print(findSpot(fileName,searchThreshold,mask))

print("done")










# fileList = fileList[:10]
# fig = plt.figure()
# plotSensitivity=3
# ims= []
# i=0
# for fileName in fileList:
#     plot_data = readLEEDImage(fileName)
#     plot_data = plot_data[450:464,422:438]
#     m, s = np.mean(plot_data), np.std(plot_data)
#     im = plt.imshow(plot_data, interpolation='nearest', cmap='jet',
#                vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
#                origin='lower')
#     ims.append([im])
#     # plt.title(str(i))
#     print(i)
#     i+=1
#
# print(len(ims))
#
# print("making animation")
# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)
# print("saving")
# ani.save('dynamic_images.mp4')
# print("ploting")
# plt.show()

