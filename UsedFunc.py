import numpy as np
from scipy.optimize import curve_fit
import sep
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import glob
import csv
rcParams['figure.figsize'] = [10., 8.]
from UsedFunc import *



def plotFunc(plot_data, plotSensitivity=3):
    m, s = np.mean(plot_data), np.std(plot_data)
    plt.imshow(plot_data, interpolation='nearest', cmap='gray', \
               vmin=m - plotSensitivity * s, \
               vmax=m + plotSensitivity * s, origin='lower')
    plt.colorbar()
    plt.show()


def readLEEDImage(filepath):
    global picWidth, picHeight
    data = np.array(Image.open(filepath))
    picWidth = len(data[1])
    picHeight = len(data)
    return data


def makeMask(mask_x_center, mask_y_center, r1, r2):
    global picWidth, picHeight
    mask = [[0 for x in range(picWidth)] for y in range(picHeight)]

    for y in range(picHeight):
        for x in range(picWidth):
            if (x - mask_x_center) ** 2 + (y - mask_y_center) ** 2 > r1 ** 2 and (x - mask_x_center) ** 2 + (
                y - mask_y_center) ** 2 < r2 ** 2:
                mask[y][x] = 1
    return np.array(mask).astype(np.uint8)


def compressImage16to8bit(imageArray, scaleFactor):
    imageArray = imageArray / scaleFactor
    #     imageArray=imageArray.astype(np.uint8)
    imageArray = imageArray
    return imageArray


def applyMask(imageArray, mask):
    appliedMask = np.multiply(imageArray, mask)
    return appliedMask


def plotSpots(imgArray, objects_list, plotSensitivity=3):
    # plot background-subtracted image
    fig, ax = plt.subplots()
    m, s = np.mean(imgArray), np.std(imgArray)
    im = ax.imshow(imgArray, interpolation='nearest', cmap='gray',
                   vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s, origin='lower')

    # plot an ellipse for each object
    for i in range(len(objects_list)):
        e = Ellipse(xy=(objects_list['x'][i], objects_list['y'][i]),
                    width=6 * objects_list['a'][i],
                    height=6 * objects_list['b'][i],
                    angle=objects_list['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)

    plt.show()


def findSpot(fileName, searchThreshold, mask, \
             scaleFactor=10, plotSensitivity=3, showSpots=False, fullInformation=False):
    imgArray = readLEEDImage(fileName)
    # plotFunc(imgArray)
    imgArray = compressImage16to8bit(imgArray, scaleFactor)
    # plotFunc(imgArray)
    imgArray = applyMask(imgArray, mask)
    # plotFunc(imgArray)

    bkg = sep.Background(imgArray)
    objects_list = sep.extract(imgArray, searchThreshold, err=bkg.globalrms)

    if showSpots == True:
        plotSpots(imgArray, objects_list, plotSensitivity)

    if fullInformation == True:
        return objects_list
    else:
        return np.array([objects_list['x'], objects_list['y']]).T


def plotFitFunc(xy, zobs, pred_params):
    x, y = xy
    xi, yi = np.mgrid[:16:30j, :16:30j]
    xyi = np.vstack([xi.ravel(), yi.ravel()])

    zpred = fitFunc(xyi, *pred_params)
    zpred.shape = xi.shape

    fig, ax = plt.subplots()
    #     ax.scatter(x, y, c=zobs, s=200, vmin=zpred.min(), vmax=zpred.max())
    im = ax.imshow(zpred, extent=[xi.min(), xi.max(), yi.max(), yi.min()], aspect='auto')
    fig.colorbar(im)
    ax.invert_yaxis()
    plt.show()