print("SEPSPA Started, Loading Libraries")
import time
import glob
import datetime
from scipy.optimize import curve_fit
import sep
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pytictoc import TicToc
import csv, itertools, json, os, shutil, ntpath
import numpy as np

class fitting:

    def __init__(self,configFilePath="configList.json"):
        self.start_time = time.time()
        self.configFilePath = configFilePath
        self.configList = json.load(open(self.configFilePath))
        self.timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        self.dataFolderName = self.configList["dataFolderName"]

        if not self.dataFolderName:
            self.fileList = glob.glob("./*.tif")
        else:
            self.fileList = glob.glob(self.dataFolderName + "/*.tif")
            self.fileList = sorted(self.fileList)

    def preStart(self):
        self.makeResultDir()
        self.CSVName = "./Result/" + self.timeStamp + "_" + self.configList["csvNameRemark"] + ".csv"
        self.copyJsontoLog()
        self.globalCounter = 0
        self.totalFileNumber = len(self.fileList)
        self.setPicDim()
        self.mask = self.makeMask()


    def makeResultDir(self):
        '''make a new directory storing fitting result if it does not exists'''
        if not os.path.exists(os.path.join(os.curdir, "Result")):
            os.makedirs(os.path.join(os.curdir, "Result"))
            print("make Result Dir")

    def copyJsontoLog(self):
        """copy the current json setting to Log dir with timestamp as name,
            create one if it does not exists"""
        if not os.path.exists(os.path.join(os.curdir, "Log")):
            os.makedirs(os.path.join(os.curdir, "Log"))
            print("make Log Dir")

        sourceDirectory = os.curdir
        newFileName = self.timeStamp + "_" + self.configList["csvNameRemark"] + ".json"
        finalDirectory = os.path.join(os.curdir, "Log")
        dstFile = os.path.join(finalDirectory, newFileName)
        sourceFile = os.path.join(sourceDirectory, "configList.json")
        shutil.copy(sourceFile, dstFile)
        print("Copied Json file to Log")

    def printSaveStatus(self):
        if self.globalCounter != 0:
            elapsedTime = ((time.time() - self.start_time) / 60)
            totalTime = elapsedTime / (self.globalCounter / fileAmount)
            timeLeft = totalTime - elapsedTime

            print("---Elapsed Time: %.2f / %.2f Minutes ---" % (elapsedTime, totalTime)
                  + "---Time Left: %.2f  Minutes ---" % timeLeft
                  + "--save to" + self.CSVName)

    def setPicDim(self):
        """init picWidth, picHeight"""
        data = np.array(Image.open(self.fileList[0]))
        self.picWidth = len(data[1])
        self.picHeight = len(data)
        print("Width: ", self.picWidth, ", Height: ", self.picHeight)
        print("Image Center: ", self.picWidth / 2, self.picHeight / 2)

    def makeMask(self):
        """create a donut shape mask with r1 as inner diameter and r2 as outer diameter"""

        mask = [[0 for x in range(self.picWidth)] for y in range(self.picHeight)]
        mask_x_center = self.configList["maskConfig"]["mask_x_center"]
        mask_y_center = self.configList["maskConfig"]["mask_y_center"]
        r1 = self.configList["maskConfig"]["innerRadius"]
        r2 = self.configList["maskConfig"]["outerRadius"]
        for y in range(self.picHeight):
            for x in range(self.picWidth):
                if (x - mask_x_center) ** 2 + (y - mask_y_center) ** 2 > r1 ** 2 and (x - mask_x_center) ** 2 + (
                        y - mask_y_center) ** 2 < r2 ** 2:
                    mask[y][x] = 1

        return np.array(mask).astype(np.uint8)








    def testMode(self):
        print("TestMode")

    def SEPMode(self):
        print("SEPMode")

    def SPAMode(self):
        print("SPAMode")