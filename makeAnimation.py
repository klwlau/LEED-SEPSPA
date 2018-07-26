import matplotlib.animation as animation
from UsedFunc import *
import glob

fig = plt.figure()
plotSensitivity=3
ims= []
startID = 0
dataFolderName = configList["dataFolderName"]
subFolder = "fitFuncFig/"
dataFolderName = dataFolderName+subFolder
fileList = glob.glob(dataFolderName + "/*.png")
fileList = sorted(fileList)
counter = 0

# fileList = fileList[:10]
for filePath in fileList:
    plot_data = readLEEDImage(filePath)
    m, s = np.mean(plot_data), np.std(plot_data)
    im = plt.imshow(plot_data, interpolation='nearest', cmap='jet',
                    vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
                    origin='lower')

    ims.append([im])
    print(counter)
    counter+=1


print("making animation")
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
print("saving animation")
ani.save('dynamic_images.avi')
print("ploting animation")
plt.show()


# import cv2
# import os
#
# image_folder = dataFolderName
# video_name = 'video.avi'
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_name, -1, 1, (width,height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()