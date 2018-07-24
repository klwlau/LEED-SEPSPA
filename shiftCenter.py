import time
import glob
start_time = time.time()
import datetime
from UsedFunc import *
from scipy.ndimage.interpolation import shift
from PIL import Image
import matplotlib.animation as animation
import ntpath


fig = plt.figure()
plotSensitivity=3
ims= []
startID = 0
endID = 1601

counter = startID
dataFolderName = configList["dataFolderName"]
fileList = glob.glob(dataFolderName + "/*.tif")
fileList = sorted(fileList)
fileList = fileList[startID:endID]
searchThreshold = 500
aniPLotRange =10
setIntCenter = False
setPicDim(fileList[0])

makeShiftCenterResultDir(dataFolderName)

mask = makeMask(configList["maskConfig"]["mask_x_center"], configList["maskConfig"]["mask_y_center"],
                0,1000)

for filePath in fileList:
    returnList,element,imageArray = findSpot(filePath, searchThreshold, mask, shiftCenterMode=True)
    xCenter,yCenter = returnList[4],returnList[5]
    xCenter, yCenter = int(xCenter), int(yCenter)
    if setIntCenter != True:
        intXCenter,intYCenter = xCenter, yCenter
        aniXUp, aniYUp = intXCenter + aniPLotRange, intYCenter + aniPLotRange
        aniXDown, aniYDown = intXCenter - aniPLotRange, intYCenter - aniPLotRange
        setIntCenter = True

    xShift = intXCenter - xCenter
    yShift = intYCenter - yCenter
    imageArray = shift(imageArray,[yShift,xShift])
    saveArray = Image.fromarray(imageArray)
    fileName = ntpath.basename(filePath)
    saveArray.save(dataFolderName+"ShiftCenterResult/"+fileName)

    plot_data = imageArray
    plot_data = plot_data[aniYDown:aniYUp, aniXDown:aniXUp]
    m, s = np.mean(plot_data), np.std(plot_data)
    im = plt.imshow(plot_data, interpolation='nearest', cmap='jet',
               vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
               origin='lower')
    ims.append([im])
    print(counter, element)
    counter+=1


print(len(ims))

print("making animation")
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
print("saving animation")
ani.save('dynamic_images.mp4')
print("ploting animation")
plt.show()
print("done")










# for fileName in fileList:
#     plot_data = readLEEDImage(fileName)
#     plot_data = plot_data[470:480,406:418]
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

