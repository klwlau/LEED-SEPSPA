import time, json, ntpath, glob, os, itertools
from pytictoc import TicToc
from scipy.ndimage.interpolation import shift
from PIL import Image
import numpy as np
import sep


class shiftCenter:
    def __init__(self, configFilePath="configList.json"):
        self.searchThreshold = 3000
        self.setIntCenter = False
        self.start_time = time.time()
        self.configFilePath = configFilePath
        self.configList = json.load(open(self.configFilePath))
        self.dataFolderName = self.configList["dataFolderName"]
        if not self.dataFolderName:
            self.fileList = glob.glob("./*.tif")
        else:
            self.fileList = glob.glob(self.dataFolderName + "/*.tif")
            self.fileList = sorted(self.fileList)
        self.setPicDim()
        self.makeShiftCenterResultDir()
        self.makeMask()

    def setPicDim(self):
        """init picWidth, picHeight"""
        data = np.array(Image.open(self.fileList[0]))
        self.picWidth = len(data[1])
        self.picHeight = len(data)
        print("Width: ", self.picWidth, ", Height: ", self.picHeight)
        print("Image Center: ", self.picWidth / 2, self.picHeight / 2)

    def makeShiftCenterResultDir(self):
        '''make a new directory storing centered LEED image if it does not exists'''
        if not os.path.exists(os.path.join(self.dataFolderName, "ShiftCenterResult")):
            os.makedirs(os.path.join(self.dataFolderName, "ShiftCenterResult"))
            print("make ShiftCenterResult Dir")

    def makeMask(self):
        """create a donut shape mask with r1 as inner diameter and r2 as outer diameter"""

        mask = [[0 for x in range(self.picWidth)] for y in range(self.picHeight)]
        mask_x_center = self.configList["maskConfig"]["mask_x_center"]
        mask_y_center = self.configList["maskConfig"]["mask_y_center"]
        r1 = 0
        r2 = 1000
        for y in range(self.picHeight):
            for x in range(self.picWidth):
                if (x - mask_x_center) ** 2 + (y - mask_y_center) ** 2 > r1 ** 2 and (x - mask_x_center) ** 2 + (
                        y - mask_y_center) ** 2 < r2 ** 2:
                    mask[y][x] = 1
        self.mask = np.array(mask).astype(np.uint8)

    def readLEEDImage(self, filePath):
        """read a image file and convert it to np array"""
        data = np.array(Image.open(filePath))
        data = np.flipud(data)
        return data

    def compressImage(self, imageArray):
        imageArray = imageArray / 1.0
        imageArray = imageArray
        return imageArray

    def applyMask(self, imageArray):
        """apply the mask to an np array"""
        appliedMask = np.multiply(imageArray, self.mask)
        return appliedMask

    def applySEPToImg(self, filePath):
        """use Sep to find rough spot location"""

        imageArray = self.readLEEDImage(filePath)
        imageArray = self.compressImage(imageArray)
        imageArray = self.applyMask(imageArray)

        bkg = sep.Background(imageArray)
        sepObjectsList = sep.extract(imageArray, self.searchThreshold, err=bkg.globalrms)
        returnList = np.array([sepObjectsList['peak'], sepObjectsList['x'], sepObjectsList['y'],
                               sepObjectsList['xmax'], sepObjectsList['ymax'],
                               sepObjectsList['a'], sepObjectsList['b'], sepObjectsList['theta']]).T

        returnList = list(itertools.chain.from_iterable(returnList))
        numberOfSpot = len(sepObjectsList)
        returnList.insert(0, numberOfSpot)

        return numberOfSpot, returnList, imageArray

    def saveImArrayTo(self, imageArray, fullPathAndFileName):
        saveArray = Image.fromarray(imageArray)
        saveArray.save(fullPathAndFileName)

    def startShiftCenter(self):

        t = TicToc()
        t.tic()
        for filePath in self.fileList:
            fileName = ntpath.basename(filePath)
            print(fileName)
            while True:
                numberOfSpot, returnList, imageArray = self.applySEPToImg(filePath)

                if numberOfSpot == 1:
                    break
                if numberOfSpot > 1:
                    self.searchThreshold = self.searchThreshold * 1.1
                if numberOfSpot < 1:
                    self.searchThreshold = self.searchThreshold * 0.9

            xCenter, yCenter = int(returnList[4]), int(returnList[5])

            if self.setIntCenter != True:
                intXCenter, intYCenter = xCenter, yCenter
                self.setIntCenter = True

            xShift = intXCenter - xCenter
            yShift = intYCenter - yCenter

            imageArray = shift(imageArray, [yShift, xShift])

            self.saveImArrayTo(imageArray, self.dataFolderName + "ShiftCenterResult/" + fileName)

        t.toc()
