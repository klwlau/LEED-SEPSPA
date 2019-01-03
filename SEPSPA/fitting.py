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
from numba import jit
import numpy as np

class fitting:

    def __init__(self,configFilePath="configList.json"):
        self.start_time = time.time()
        self.configFilePath = configFilePath
        self.configList = json.load(open(self.configFilePath))
        self.timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        #loading confingList
        self.dataFolderName = self.configList["dataFolderName"]
        self.cropRange = self.configList["findSpotParameters"]["cropRange"]
        self.searchThreshold = self.configList["findSpotParameters"]["searchThreshold"]
        self.guessUpBound = self.configList["fittingParameters"]["guessUpBound"]
        self.intConfigGuess = self.configList["fittingParameters"]["intGuess"]
        self.guessLowBound = self.configList["fittingParameters"]["guessLowBound"]
        self.dataFolderName = self.configList["dataFolderName"]
        self.plotSensitivity_low = self.configList["testModeParameters"]["plotSensitivity_low"]
        self.plotSensitivity_up = self.configList["testModeParameters"]["plotSensitivity_up"]

        if not self.dataFolderName:
            self.fileList = glob.glob("./*.tif")
        else:
            self.fileList = glob.glob(self.dataFolderName + "/*.tif")
            self.fileList = sorted(self.fileList)
        self.CSVwriteBuffer = self.configList["CSVwriteBuffer"]



    def preStart(self):
        self.makeResultDir()
        self.CSVName = "./Result/" + self.timeStamp + "_" + self.configList["csvNameRemark"] + ".csv"
        self.copyJsontoLog()
        self.globalCounter = 0
        self.totalFileNumber = len(self.fileList)
        self.setPicDim()
        self.makeMask()
        self.copyJsontoLog()
        self.makeResultDir()


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

    def readLEEDImage(self,filePath):
        """read a image file and convert it to np array"""
        data = np.array(Image.open(filePath))
        data = np.flipud(data)
        return data

    def printSaveStatus(self):
        if self.globalCounter != 0:
            elapsedTime = ((time.time() - self.start_time) / 60)
            totalTime = elapsedTime / (self.globalCounter / self.totalFileNumber)
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
        self.mask= np.array(mask).astype(np.uint8)

    def applyMask(self,imageArray):
        """apply the mask to an np array"""
        appliedMask = np.multiply(imageArray, self.mask)
        return appliedMask

    def plotSpots(self,imgArray, objects_list,
                  saveMode=False, saveFileName="test", showSpots=False):
        """plot sep result"""
        fig, ax = plt.subplots()
        min_int, max_int = np.amin(imgArray), np.amax(imgArray)
        plt.imshow(imgArray, interpolation='nearest', cmap='jet',
                   vmin=min_int + (max_int - min_int) * self.plotSensitivity_low,
                   vmax=min_int + (max_int - min_int) * self.plotSensitivity_up,
                   origin='lower')

        # plot an ellipse for each object
        for i in range(len(objects_list)):
            e = Ellipse(xy=(objects_list['x'][i], objects_list['y'][i]),
                        width=3 * objects_list['a'][i],
                        height=3 * objects_list['b'][i],
                        angle=objects_list['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

        plt.colorbar()
        if saveMode:
            savePath = self.configList["saveFigModeParameters"]["saveFigFolderName"]
            plt.savefig(savePath + saveFileName + ".png")

        if showSpots:
            plt.show()
        else:
            plt.clf()

    # def checkShowSpots(self):
    #     if self.show

    @jit
    def getSpotRoughRange(self,imgArray: np.array,
                          showSpots: bool = False, fittingMode: bool = False, saveMode=False, printReturnArray=False,
                          saveFileName="test"):

        imgArray = self.applyMask(imgArray)

        bkg = sep.Background(imgArray)
        sepObjectsList = sep.extract(imgArray, self.searchThreshold, err=bkg.globalrms)

        if showSpots is True or saveMode is True:
            self.plotSpots(imgArray, sepObjectsList,
                      showSpots=showSpots, saveMode=saveMode, saveFileName=saveFileName)

        if fittingMode is True:
            returnArray = np.array([sepObjectsList['xcpeak'], sepObjectsList['ycpeak']]).T

            if printReturnArray:
                print(len(returnArray))
                print(returnArray)
            return returnArray, sepObjectsList

        else:
            returnArray = np.array([sepObjectsList['peak'], sepObjectsList['x'], sepObjectsList['y'],
                                    sepObjectsList['xmax'], sepObjectsList['ymax'],
                                    sepObjectsList['a'], sepObjectsList['b'], sepObjectsList['theta']]).T
            if printReturnArray:
                print(len(returnArray))
                print(returnArray)
            return returnArray,sepObjectsList







    def testMode(self):
        print("TestMode")

    def SEPMode(self):
        self.preStart()
        self.SEPDict ={}
        print("SEPMode")
        writeBufferArray =[]
        FileHeader=["FileID", "File Name", "Number of Spots"]
        SEPparameterArrray=["Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta"]

        for filePath in self.fileList:
            imageArray = self.readLEEDImage(filePath)
            z,b = self.getSpotRoughRange(imageArray)
            print(z)
            print(b)




        print("save to :" + self.CSVName)
        return self.SEPDict

    def SPAMode(self):
        print("SPAMode")
        print("save to :" + self.CSVName)