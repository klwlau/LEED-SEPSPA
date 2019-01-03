import time
import glob

start_time = time.time()
from UsedFunc import *
from scipy.ndimage.interpolation import shift
import matplotlib.animation as animation
import ntpath

fig = plt.figure()
plotSensitivity = 3
ims = []
startID = 0
dataFolderName = configList["dataFolderName"]
fileList = glob.glob(dataFolderName + "/*.tif")
fileList = sorted(fileList)
makeAnimation = False
searchThreshold = 1000
aniPLotRange = 10

counter = startID
setPicDim(fileList[0])
makeShiftCenterResultDir(dataFolderName)

mask = makeMask(configList["maskConfig"]["mask_x_center"], configList["maskConfig"]["mask_y_center"],
                0, 1000)
errorList = []
setIntCenter = False
for filePath in fileList:
    while True:
        returnList, element, imageArray = findSpot(filePath, searchThreshold, mask, shiftCenterMode=True)
        if element == 1:
            break
        if element > 1:
            searchThreshold = searchThreshold * 1.1
        if element < 1:
            searchThreshold = searchThreshold * 0.9
        print("repeat ", counter, ", element: ", element, ", new searchThreshold: ", searchThreshold)

    xCenter, yCenter = returnList[4], returnList[5]
    xCenter, yCenter = int(xCenter), int(yCenter)
    if setIntCenter != True:
        intXCenter, intYCenter = xCenter, yCenter
        aniXUp, aniYUp = intXCenter + aniPLotRange, intYCenter + aniPLotRange
        aniXDown, aniYDown = intXCenter - aniPLotRange, intYCenter - aniPLotRange
        setIntCenter = True

    xShift = intXCenter - xCenter
    yShift = intYCenter - yCenter
    imageArray = shift(imageArray, [yShift, xShift])
    fileName = ntpath.basename(filePath)

    saveImArrayTo(imageArray, dataFolderName + "ShiftCenterResult/" + fileName)

    plot_data = imageArray
    plot_data = plot_data[aniYDown:aniYUp, aniXDown:aniXUp]
    m, s = np.mean(plot_data), np.std(plot_data)
    im = plt.imshow(plot_data, interpolation='nearest', cmap='jet',
                    vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
                    origin='lower')
    ims.append([im])

    if element != 1:
        errorList.append(counter)

    print(counter, element)
    counter += 1

if makeAnimation:
    print("making animation")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    print("saving animation")
    ani.save('dynamic_images.mp4')
    print("ploting animation")
    plt.show()

print("errorList: ", errorList)
print("done")