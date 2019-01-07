print("Loading Libraries")
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
from joblib import Parallel, delayed


class fitting:

    def __init__(self, configFilePath="configList.json"):
        self.start_time = time.time()
        self.configFilePath = configFilePath
        self.configList = json.load(open(self.configFilePath))
        self.timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        # loading confingList
        self.dataFolderName = self.configList["dataFolderName"]
        self.cropRange = self.configList["SEPParameters"]["cropRange"] // 2
        self.searchThreshold = self.configList["SEPParameters"]["searchThreshold"]
        self.guessUpBound = self.configList["SPAParameters"]["guessUpBound"]
        # self.intConfigGuess = self.configList["SPAParameters"]["intGuess"]
        self.guessLowBound = self.configList["SPAParameters"]["guessLowBound"]
        self.dataFolderName = self.configList["dataFolderName"]
        self.sepPlotColourMin = self.configList["testModeParameters"]["sepPlotColourMin"]
        self.sepPlotColourMax = self.configList["testModeParameters"]["sepPlotColourMax"]
        self.saveSEPResult = self.configList["SEPParameters"]["saveSEPResult"]
        self.scaleDownFactor = self.configList["SEPParameters"]["scaleDownFactor"]

        if not self.dataFolderName:
            self.fileList = glob.glob("./*.tif")
        else:
            self.fileList = glob.glob(self.dataFolderName + "/*.tif")
            self.fileList = sorted(self.fileList)
        self.CSVwriteBuffer = self.configList["CSVwriteBuffer"]
        self.preStart()
        self.maxSpotInFrame = 0
        self.fittingBoundDict = {}
        self.fittingIntDict = {}

    def preStart(self):
        self.makeResultDir()
        self.SEPCSVName = "./Result/" + self.timeStamp + "_" + self.configList["csvNameRemark"] + "_SEP.csv"
        self.SPACSVName = "./Result/" + self.timeStamp + "_" + self.configList["csvNameRemark"] + "_SPA.csv"
        self.copyJsontoLog()
        self.globalCounter = 0
        self.totalFileNumber = len(self.fileList)
        self.setPicDim()
        self.makeMask()
        self.copyJsontoLog()
        self.makeResultDir()
        if self.saveSEPResult:
            self.makeDirInDataFolder("SEPResult")
        self.sepComplete = False

    def saveToCSV(self, RowArray, fileName):
        """save a list of row to CSV file"""
        with open(fileName, 'a', newline='') as f:
            csvWriter = csv.writer(f)
            for i in RowArray:
                csvWriter.writerow(i)
        print("save to :" + fileName)

    def makeResultDir(self):
        '''make a new directory storing fitting result if it does not exists'''
        if not os.path.exists(os.path.join(os.curdir, "Result")):
            os.makedirs(os.path.join(os.curdir, "Result"))
            print("make Result Dir")

    def makeDirInDataFolder(self, dirName):
        '''make a new directory with dirName if it does not exists'''
        if not os.path.exists(os.path.join(self.dataFolderName, dirName)):
            os.makedirs(os.path.join(self.dataFolderName, dirName))
            print("make ", dirName, " Dir")
        return os.path.join(self.dataFolderName, dirName)

    def saveDictToPLK(self, dict, fileName):
        """pickle an dict to a locaiton"""
        import pickle
        dirPath = self.makeDirInDataFolder("pythonObj") + "/"
        with open(dirPath + fileName + '.pkl', 'wb') as f:
            pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

    def readPLK(self, filePath):
        """load a pickle object"""
        import pickle
        with open(filePath, 'rb') as f:
            return pickle.load(f)

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

    def readLEEDImage(self, filePath):
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
                  + "--save to" + self.SEPCSVName)

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
        self.mask = np.array(mask).astype(np.uint8)

    def compressImage(self, imageArray):
        imageArray = imageArray / self.scaleDownFactor
        imageArray = imageArray
        return imageArray

    def applyMask(self, imageArray):
        """apply the mask to an np array"""
        appliedMask = np.multiply(imageArray, self.mask)
        return appliedMask

    def genFittingBound(self, numOfGauss=1):
        numOfGaussKey = str(numOfGauss)
        if numOfGaussKey in self.fittingBoundDict:
            return self.fittingBoundDict[numOfGaussKey]
        else:
            guessUpBound = self.configList["SPAParameters"]["backgroundGuessUpperBound"]
            guessLowBound = self.configList["SPAParameters"]["backgroundGuessLowerBound"]

            for num in range(numOfGauss):
                guessUpBound += self.configList["SPAParameters"]["gaussianUpperBoundTemplate"]
                guessLowBound += self.configList["SPAParameters"]["gaussianLowerBoundTemplate"]

            self.fittingBoundDict[numOfGaussKey] = [guessLowBound, guessUpBound]

            return self.fittingBoundDict[numOfGaussKey]

    def genIntCondittion(self, sepSpotDict, numOfGauss=1):
        intGuess = self.configList["SPAParameters"]["backgroundIntGuess"]

        for i in range(numOfGauss):
            if i == 0:
                intGuess.append(sepSpotDict["Am"])
                intGuess.append(self.cropRange / 2)
                intGuess.append(self.cropRange / 2)
                intGuess.append(sepSpotDict["a"])
                intGuess.append(sepSpotDict["b"])
                intGuess.append(sepSpotDict["theta"])
            else:
                intGuess.append(self.configList["SPAParameters"]["minorGaussianIntGuess"])

        return intGuess

    def plotSEPReult(self, imgArray, objects_list,
                     saveMode=False, saveFileName="test", showSpots=False):
        """plot sep result"""
        fig, ax = plt.subplots()
        min_int, max_int = np.amin(imgArray), np.amax(imgArray)
        # plt.imshow(imgArray, interpolation='nearest', cmap='jet',
        #            vmin=min_int + (max_int - min_int) * self.plotSensitivity_low,
        #            vmax=min_int + (max_int - min_int) * self.plotSensitivity_up
        #            , origin='lower')

        plt.imshow(imgArray, interpolation='nearest', cmap='jet',
                   vmin=self.sepPlotColourMin, vmax=self.sepPlotColourMax,
                   origin='lower')

        """plot an ellipse for each object"""
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
            # plt.show()
            saveDir = self.dataFolderName + "SEPResult/"
            plt.savefig(saveDir + saveFileName + ".jpg", dpi=500)

        if showSpots:
            plt.show()
        else:
            plt.close()

    @jit
    def applySEPToImg(self, imgArray: np.array):
        """use Sep to find rough spot location"""

        imgArray = self.applyMask(imgArray)

        bkg = sep.Background(imgArray)
        sepObjectsList = sep.extract(imgArray, self.searchThreshold, err=bkg.globalrms)
        returnList = np.array([sepObjectsList['peak'], sepObjectsList['x'], sepObjectsList['y'],
                               sepObjectsList['xmax'], sepObjectsList['ymax'],
                               sepObjectsList['a'], sepObjectsList['b'], sepObjectsList['theta']]).T

        returnList = list(itertools.chain.from_iterable(returnList))
        returnList.insert(0, len(sepObjectsList))

        return sepObjectsList, returnList

    def appendSepObjectIntoSEPDict(self, fileID, filePath, sepObject):
        frameDict = {}

        for spotID, spot in enumerate(sepObject):
            tempSpotDict = {}
            tempSpotDict["Am"] = spot['peak']
            tempSpotDict["x"] = spot['x']
            tempSpotDict["y"] = spot['y']
            tempSpotDict["xmax"] = spot['xmax']
            tempSpotDict["ymax"] = spot['ymax']
            tempSpotDict["xcpeak"] = spot['xcpeak']
            tempSpotDict["ycpeak"] = spot['ycpeak']
            tempSpotDict["a"] = spot['a']
            tempSpotDict["b"] = spot['b']
            tempSpotDict["theta"] = spot['theta']
            frameDict[str(spotID)] = tempSpotDict

        frameDict["filePath"] = filePath
        frameDict["numberOfSpot"] = len(sepObject)
        self.sepDict[str(fileID)] = frameDict
        self.maxSpotInFrame = max(frameDict["numberOfSpot"], self.maxSpotInFrame)
        self.sepDict["maxSpotInFrame"] = self.maxSpotInFrame

    def testMode(self):
        print("TestMode")

    def sepMode(self):

        def parallelSEP(fileID, filePath):
            imageArray = self.readLEEDImage(filePath)
            imageArray = self.compressImage(imageArray)
            imageArray = self.applyMask(imageArray)

            # imageArray = self.applyMask(imageArray)
            sepObject, sepWriteCSVList = self.applySEPToImg(imageArray)
            sepWriteCSVList.insert(0, filePath)
            sepWriteCSVList.insert(0, fileID)

            if self.saveSEPResult:
                self.plotSEPReult(imageArray, sepObject, saveMode=True,
                                  saveFileName=os.path.basename(filePath)[:-4] + "_SEP")

            return (sepObject, sepWriteCSVList)

        print("SEPMode Start")
        time.sleep(0.1)
        self.sepDict = {}
        sepCSVHeader = ["FileID", "File Name", "Number of Spots"]
        SEPparameterHeader = ["Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta"]

        for i in range(15):
            sepCSVHeader += SEPparameterHeader

        self.saveToCSV([sepCSVHeader], self.SEPCSVName)

        if self.configList["singleCoreDebugMode"] != True:
            with Parallel(n_jobs=-1, verbose=1) as parallel:
                multicoreSEP = parallel(
                    delayed(parallelSEP)(fileID, filePath) for fileID, filePath in enumerate(self.fileList))
        else:
            multicoreSEP = []
            for fileID, filePath, in enumerate(self.fileList):
                multicoreSEP.append(parallelSEP(fileID, filePath))

        writeBufferArray = []

        for fileID, i in enumerate(multicoreSEP):
            writeBufferArray.append(i[1])
            filePath = i[1][1]
            self.appendSepObjectIntoSEPDict(fileID, filePath, i[0])

        self.saveToCSV(writeBufferArray, self.SEPCSVName)
        self.saveDictToPLK(self.sepDict, self.timeStamp + "_" + self.configList["csvNameRemark"] + "_SEPDict")

        self.sepComplete = True
        print("SEPMode Complete")
        return self.sepDict

    def spaMode(self):
        print("SPAMode")
        if self.sepComplete == False:
            print("Runing SEPMode to get Rough range")
            self.sepMode()

        for frameID, frameDict in self.sepDict.items():
            if type(frameDict) is dict:
                numberOfSpot = frameDict["numberOfSpot"]
                frameFilePath = frameDict["filePath"]
                imageArray = self.readLEEDImage(frameFilePath)

                for spotID in range(numberOfSpot):
                    xyzArray = []
                    sepSpotDict = frameDict[str(spotID)]
                    cropedArray = imageArray[
                                  int(sepSpotDict["ycpeak"]) - self.cropRange: int(
                                      sepSpotDict["ycpeak"]) + self.cropRange,
                                  int(sepSpotDict["xcpeak"]) - self.cropRange: int(
                                      sepSpotDict["xcpeak"]) + self.cropRange]
                    for xx in range(len(cropedArray)):
                        for yy in range(len(cropedArray[xx])):
                            xyzArray.append([xx, yy, cropedArray[xx][yy]])

                    x, y, z = np.array(xyzArray).T
                    xy = x, y



                    # intGuess = self.genIntCondittion(sepSpotDict)
                    # fittingBound  =self.genFittingBound()

                    # plt.imshow(cropedArray)
                    # plt.show()

                    # print(sepSpotDict)

        print("save to :" + self.SPACSVName)
