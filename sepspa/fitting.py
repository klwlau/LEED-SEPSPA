import time
import glob
import datetime
from scipy.optimize import curve_fit
import sep
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pytictoc import TicToc
import csv, itertools, json, os, shutil
from numba import jit
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import dblquad
import sepspa.fitFunc as fitFunc
from matplotlib.colors import LinearSegmentedColormap


class fitting:

    def __init__(self, configFilePath="configList.json", listLength="Full"):
        self.listLength = listLength
        self.start_time = time.time()
        self.setDimStatus = True
        self.configFilePath = configFilePath
        np.set_printoptions(precision=3, suppress=True)
        self.timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        self.preStart()
        self.copyJsontoLog()
        self.totalFileNumber = len(self.fileList)
        self.csvHeaderLength = 15
        self.fittingBoundDict = {}
        self.fittingIntDict = {}

    def preStart(self):
        self.makeResultDir()
        self.configDict = json.load(open(self.configFilePath))
        self.dataFolderName = self.configDict["dataFolderName"]
        self.halfCropRange = self.configDict["SEPParameters"]["cropRange"] // 2
        self.searchThreshold = self.configDict["SEPParameters"]["searchThreshold"]
        self.dataFolderName = self.configDict["dataFolderName"]
        self.sepPlotColourMin = self.configDict["testModeParameters"]["sepPlotColourMin"]
        self.sepPlotColourMax = self.configDict["testModeParameters"]["sepPlotColourMax"]
        self.saveSEPResult = self.configDict["SEPParameters"]["saveSEPResult"]
        self.scaleDownFactor = self.configDict["SEPParameters"]["scaleDownFactor"]
        self.CSVwriteBuffer = self.configDict["CSVwriteBuffer"]
        self.multipleSpotInFrameThreshold = self.configDict["SPAParameters"]["multipleSpotInFrameRange"] / 2
        self.SEPCSVName = "./Result/" + self.timeStamp + "_" + self.configDict["csvRemark"] + "_SEP.csv"
        self.SPACSVNameRaw = "./Result/" + self.timeStamp + "_" + self.configDict["csvRemark"] + "_SPARaw.csv"
        self.SPACSVNameEllipticalCorrected = "./Result/" + self.timeStamp + "_" + self.configDict[
            "csvRemark"] + "_SPAEllipticalCorrected.csv"

        self.globalCounter = 0

        if not self.dataFolderName:
            self.fileList = glob.glob("./*.tif")
        else:
            self.fileList = glob.glob(self.dataFolderName + "/*.tif")
        self.fileList = sorted(self.fileList)
        if self.listLength != "Full":
            self.fileList = self.fileList[:self.listLength]

        if self.setDimStatus:
            self.setPicDim()
            self.setDimStatus = False

        self.makeMask()
        self.makeResultDir()
        if self.saveSEPResult:
            self.makeDirInDataFolder("SEPResult")
        self.sepComplete = False

        self.SPACSVHeader = ["FileID", "File Path", "Number of Spots", "Fitting Time"]
        self.SPAparameterHeader = ["Am", "x", "y", "sigma_x", "sigma_y", "theta", "A", "B", "Constant"]

    def saveToCSV(self, writeArray, fileName):
        """save a list of row to CSV file"""
        with open(fileName, 'a', newline='') as f:
            csvWriter = csv.writer(f)
            for row in writeArray:
                csvWriter.writerow(row)
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

    def loadPLK(self, filePath):
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
        newFileName = self.timeStamp + "_" + self.configDict["csvRemark"] + ".json"
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
        # print(self.fileList)
        data = np.array(Image.open(self.fileList[0]))
        self.picWidth = len(data[1])
        self.picHeight = len(data)
        print("Width: ", self.picWidth, ", Height: ", self.picHeight)
        print("Image Center: ", self.picWidth / 2, self.picHeight / 2)

    def makeMask(self):
        """create a donut shape mask with r1 as inner diameter and r2 as outer diameter"""

        mask = [[0 for x in range(self.picWidth)] for y in range(self.picHeight)]
        mask_x_center = self.configDict["maskConfig"]["mask_x_center"]
        mask_y_center = self.configDict["maskConfig"]["mask_y_center"]
        r1 = self.configDict["maskConfig"]["innerRadius"]
        r2 = self.configDict["maskConfig"]["outerRadius"]
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

    def genIntCondittion(self, spotID, frameID, sepSpotDict, numOfGauss=1):
        intGuess = self.configDict["SPAParameters"]["backgroundIntGuess"].copy()

        for i in range(numOfGauss):
            if i == 0:
                intGuess += [sepSpotDict["Am"]]
                intGuess += [self.halfCropRange]
                intGuess += [self.halfCropRange]
                intGuess += [sepSpotDict["a"]]
                intGuess += [sepSpotDict["b"]]
                intGuess += [sepSpotDict["theta"]]
            else:
                if self.configDict["SPAParameters"]["smartConfig"]:
                    tempMinorGaussianIntGuess = self.configDict["SPAParameters"]["minorGaussianIntGuess"].copy()
                    tempMinorGaussianIntGuess[2] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][0]
                    tempMinorGaussianIntGuess[1] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][1]
                    intGuess += tempMinorGaussianIntGuess
                else:
                    intGuess += self.configDict["SPAParameters"]["minorGaussianIntGuess"]

        return intGuess

    def genFittingBound(self, spotID, frameID, numOfGauss=1):
        guessUpBound = self.configDict["SPAParameters"]["backgroundGuessUpperBound"].copy()
        guessLowBound = self.configDict["SPAParameters"]["backgroundGuessLowerBound"].copy()

        for i in range(numOfGauss):
            tempSpotUpBound = self.configDict["SPAParameters"]["gaussianUpperBoundTemplate"].copy()
            tempSpotLowBound = self.configDict["SPAParameters"]["gaussianLowerBoundTemplate"].copy()
            if i == 0:
                tempSpotUpBound[2] = self.halfCropRange + self.configDict["SPAParameters"]["majorGaussianXYRange"]
                tempSpotUpBound[1] = self.halfCropRange + self.configDict["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[2] = self.halfCropRange - self.configDict["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[1] = self.halfCropRange - self.configDict["SPAParameters"]["majorGaussianXYRange"]
            else:
                tempSpotUpBound[2] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][0] + \
                                     self.configDict["SPAParameters"]["majorGaussianXYRange"]
                tempSpotUpBound[1] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][1] + \
                                     self.configDict["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[2] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][0] - \
                                      self.configDict["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[1] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][1] - \
                                      self.configDict["SPAParameters"]["majorGaussianXYRange"]
            guessUpBound += tempSpotUpBound
            guessLowBound += tempSpotLowBound
        return [guessLowBound, guessUpBound]

    def calRSquareError(self, fittedArray, rawArray):
        """calculate R Square error"""
        errorSquare = (fittedArray - rawArray) ** 2
        SS_tot = np.sum((fittedArray - np.mean(fittedArray)) ** 2)
        SS_res = np.sum(errorSquare)
        return (SS_res / SS_tot)

    def genSEPReultPlot(self, imgArray, objects_list,
                        saveMode=False, saveFileName="test", showPlot=False):
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
            saveDir = self.dataFolderName + "/SEPResult/"
            plt.savefig(saveDir + saveFileName + ".png", dpi=500)

        if showPlot:
            plt.show()
        else:
            plt.close()

    @jit
    def applySEPToImg(self, imgArray: np.array):
        """use Sep to find rough spot location"""

        # imgArray = self.applyMask(imgArray)

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
        # self.maxSpotInFrame = max(frameDict["numberOfSpot"], self.maxSpotInFrame)
        # self.sepDict["maxSpotInFrame"] = self.maxSpotInFrame

    def testMode(self):
        print("TestMode")
        """read json and set parameters again"""
        self.preStart()
        testModeConfigDict = self.configDict["testModeParameters"]
        """run sep"""
        testModeFileID = testModeConfigDict["testModeFileID"]
        self.sepCore(testModeFileID, self.fileList[testModeFileID], plotSEPResult=testModeConfigDict["showSpots"])
        """run spa"""

    def sepCore(self, fileID, filePath, plotSEPResult=False):

        imageArray = self.readLEEDImage(filePath)
        imageArray = self.compressImage(imageArray)
        imageArray = self.applyMask(imageArray)

        sepObject, sepWriteCSVList = self.applySEPToImg(imageArray)
        sepWriteCSVList.insert(0, filePath)
        sepWriteCSVList.insert(0, fileID)

        if self.saveSEPResult:
            self.genSEPReultPlot(imageArray, sepObject, saveMode=True,
                                 saveFileName=os.path.basename(filePath)[:-4] + "_SEP")

        if plotSEPResult:
            self.genSEPReultPlot(imageArray, sepObject, showPlot=True)

        return (sepObject, sepWriteCSVList)

    def sepMode(self):

        print("SEPMode Start")
        time.sleep(0.1)
        self.sepDict = {}
        sepCSVHeader = ["FileID", "File Path", "Number of Spots"]
        SEPparameterHeader = ["Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta"]

        for i in range(self.csvHeaderLength):
            sepCSVHeader += SEPparameterHeader

        self.saveToCSV([sepCSVHeader], self.SEPCSVName)

        if self.configDict["sepSingleCoreDebugMode"] != True:
            with Parallel(n_jobs=-1, verbose=2) as parallel:
                multicoreSEP = parallel(
                    delayed(self.sepCore)(fileID, filePath) for fileID, filePath in enumerate(self.fileList))
        else:
            multicoreSEP = []
            for fileID, filePath, in enumerate(self.fileList):
                multicoreSEP.append(self.sepCore(fileID, filePath))

        writeBufferArray = []

        for fileID, i in enumerate(multicoreSEP):
            writeBufferArray.append(i[1])
            filePath = i[1][1]
            self.appendSepObjectIntoSEPDict(fileID, filePath, i[0])

        self.saveToCSV(writeBufferArray, self.SEPCSVName)
        self.saveDictToPLK(self.sepDict, self.timeStamp + "_" + self.configDict["csvRemark"] + "_SEPDict")

        self.createNGaussDict()
        print("SEPMode Complete")
        self.sepComplete = True
        return self.sepDict

    def createNGaussDict(self):
        """reutrn a dict storing how many Gaussian needed for each spot crop"""
        print("Creating NGauss Dict")
        self.genNGaussDict = {}
        self.neighborSpotDict = {}

        for frameID, frameDict in self.sepDict.items():
            frameGaussCount = {}
            neighborFrameDict = {}
            if type(frameDict) is dict:
                numberOfSpot = frameDict["numberOfSpot"]
                for spotIID in range(numberOfSpot):
                    gaussCount = 1
                    neighborSpotList = []
                    for spotJID in range(numberOfSpot):  # for spotJID in range(spotIID, numberOfSpot):
                        if spotIID != spotJID:
                            spotI = np.array([frameDict[str(spotIID)]["xcpeak"], frameDict[str(spotIID)]["ycpeak"]])
                            spotJ = np.array([frameDict[str(spotJID)]["xcpeak"], frameDict[str(spotJID)]["ycpeak"]])
                            if spotI[0] - self.multipleSpotInFrameThreshold <= spotJ[0] <= spotI[
                                0] + self.multipleSpotInFrameThreshold and \
                                    spotI[1] - self.multipleSpotInFrameThreshold <= spotJ[1] <= spotI[
                                1] + self.multipleSpotInFrameThreshold:
                                gaussCount += 1
                                neighborSpotList.append(spotJ - (
                                        spotI - [self.halfCropRange, self.halfCropRange]))
                        if len(neighborSpotList) > 0:
                            neighborFrameDict[str(spotIID)] = neighborSpotList

                    frameGaussCount[str(spotIID)] = gaussCount
                self.genNGaussDict[str(frameID)] = frameGaussCount
                self.neighborSpotDict[str(frameID)] = neighborFrameDict

        return self.genNGaussDict

    def genFittedFuncArray(self, fit_params, outputZpredOnly=False):
        """generate an image array from the fitted function"""
        fullRange = self.halfCropRange * 2
        xi, yi = np.mgrid[0:fullRange, 0:fullRange]

        xyi = np.vstack([xi.ravel(), yi.ravel()])
        numOfGauss = int((len(fit_params) - 3) / len(self.configDict["SPAParameters"]["gaussianUpperBoundTemplate"]))

        zpred = fitFunc.NGauss(numOfGauss)(xyi, *fit_params)

        zpred.shape = xi.shape

        if outputZpredOnly:
            return zpred
        else:
            return xi, yi, zpred

    def plotFitFunc(self, fit_params, cropedRawDataArray, plotSensitivity=5, saveFitFuncPlot=False,
                    saveFitFuncFileName="fitFuncFig", plottitle="", figTxt=""):

        # Chi_square = fit_params[-1]
        # fit_params = fit_params[:-1]

        xi, yi, zpred = self.genFittedFuncArray(fit_params)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(7, 8, forward=True)

        m, s = np.mean(cropedRawDataArray), np.std(cropedRawDataArray)
        cs = ax1.imshow(cropedRawDataArray, interpolation='nearest', cmap='jet',
                        vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
                        origin='lower')

        fig.colorbar(cs)
        # plt.title("Chi^2= %.2f" % (Chi_square))
        plt.title(plottitle)
        fig.text(.5, 0.05, figTxt, ha='center')
        ax1.contour(yi, xi, zpred,
                    vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s, alpha=1, origin='lower')  # cmap='jet',
        if saveFitFuncPlot:
            if saveFitFuncFileName == "fitFuncFig":
                plt.savefig(saveFitFuncFileName + ".png")
            else:
                saveFigFullPath = self.makeDirInDataFolder("fitFuncFig_"
                                                           + self.configDict["SPAParameters"][
                                                               "saveFitFuncPlotFileRemark"])
                plt.savefig(saveFigFullPath + "/" + saveFitFuncFileName + ".png")
            plt.close(fig)
            return

        plt.show()

    def saveSpotCropFig(self, imageArray, numOfGauss, fileName="test", dirName="spotCrop"):
        self.makeDirInDataFolder(dirName)
        saveDir = self.dataFolderName + dirName + "/"
        plt.imshow(imageArray)
        plt.title(numOfGauss)
        plt.savefig(saveDir + fileName + ".png", dpi=500)
        plt.close()

    def convertSPADictIntoCSVWriteArray(self, SPADict):
        CSVWriteArray = []
        for frameID in range(len(SPADict)):
            frameDict = SPADict[str(frameID)]
            frameWriteArray = []
            spotArray = []
            frameWriteArray.append(frameID)
            frameWriteArray.append(frameDict["filePath"])
            frameWriteArray.append(frameDict["numberOfSpot"])
            frameWriteArray.append(frameDict["FittingTime"])

            for spotID in range(frameDict["numberOfSpot"]):
                spotArray.append(frameDict[str(spotID)]["Am"])
                spotArray.append(frameDict[str(spotID)]["xCenter"])
                spotArray.append(frameDict[str(spotID)]["yCenter"])
                spotArray.append(frameDict[str(spotID)]["sigma_x"])
                spotArray.append(frameDict[str(spotID)]["sigma_y"])
                spotArray.append(frameDict[str(spotID)]["theta"])
                spotArray.append(frameDict[str(spotID)]["A"])
                spotArray.append(frameDict[str(spotID)]["B"])
                spotArray.append(frameDict[str(spotID)]["C"])

            # spotArray = list(itertools.chain.from_iterable(spotArray))
            frameWriteArray += spotArray

            CSVWriteArray.append(frameWriteArray)

        return CSVWriteArray

    def sepspaMode(self, sepDict=None):

        SPAFrameTimer = TicToc()
        SPATimer = TicToc()

        def genPlotTxt(fit_para):
            """gen a string that print under the plot"""
            returnTxt = "Background: "
            returnTxt += str(fit_para[:3]) + "\n"
            for i in range(len(fit_para[3:]) // 6):
                returnTxt += "Gauss_" + str(i) + ": "
                returnTxt += str(fit_para[i * 6 + 3:i * 6 + 6 + 3])
                returnTxt += "\n"
            return returnTxt

        self.chiSqPlotList = []

        def applySPA(frameID, frameDict):

            SPAFrameTimer.tic()

            if int(frameID) % 50 == 0:
                print("Fitting Frame ID:", frameID, end=', ')
                SPATimer.toc()
            if type(frameDict) is dict:
                fitParamsFrameDict = {}
                fitUncertDict = {}
                numberOfSpot = frameDict["numberOfSpot"]
                frameFilePath = frameDict["filePath"]
                imageArray = self.readLEEDImage(frameFilePath)

                for spotID in range(numberOfSpot):
                    if self.configDict["SPAParameters"]["adaptiveGaussianFitting"]:
                        numOfGauss = self.genNGaussDict[str(frameID)][str(spotID)]
                    else:
                        numOfGauss = 1

                    xyzArray = []
                    sepSpotDict = frameDict[str(spotID)]
                    cropedArray = imageArray[
                                  int(sepSpotDict["ycpeak"]) - self.halfCropRange: int(
                                      sepSpotDict["ycpeak"]) + self.halfCropRange,
                                  int(sepSpotDict["xcpeak"]) - self.halfCropRange: int(
                                      sepSpotDict["xcpeak"]) + self.halfCropRange]
                    for xx in range(len(cropedArray)):
                        for yy in range(len(cropedArray[xx])):
                            xyzArray.append([xx, yy, cropedArray[xx][yy]])

                    xi, yi, z = np.array(xyzArray).T
                    xyi = xi, yi

                    # self.saveSpotCropFig(cropedArray,numOfGauss,fileName=os.path.basename(frameFilePath)[:-4]+"_"+str(spotID))

                    intGuess = self.genIntCondittion(spotID, frameID, sepSpotDict, numOfGauss=numOfGauss)
                    fittingBound = self.genFittingBound(spotID, frameID, numOfGauss=numOfGauss)

                    try:
                        fit_params, uncert_cov = curve_fit(fitFunc.NGauss(numOfGauss), xyi, z, p0=intGuess,
                                                           bounds=fittingBound)
                    except RuntimeError:
                        self.saveSpotCropFig(cropedArray, numOfGauss,
                                             fileName=os.path.basename(frameFilePath)[:-4] + "_" + str(spotID),
                                             dirName="runTimeError")
                        numOfGauss = 1
                        print("Runtime error, set numOfGauss = 1")
                        fit_params, uncert_cov = curve_fit(fitFunc.NGauss(numOfGauss), xyi, z, p0=intGuess,
                                                           bounds=fittingBound)

                    rSquare = self.calRSquareError(self.genFittedFuncArray(fit_params, outputZpredOnly=True),
                                                   cropedArray)

                    if self.configDict["SPAParameters"]["saveFitFuncPlot"]:
                        self.plotFitFunc(fit_params, cropedArray, saveFitFuncPlot=True,
                                         saveFitFuncFileName=os.path.basename(frameFilePath)[:-4] + "_" + str(spotID),
                                         plottitle=str(numOfGauss) + "_" + str(rSquare),
                                         figTxt=genPlotTxt(fit_params))

                    self.chiSqPlotList.append(rSquare)

                    """coordinate transformation"""
                    fit_params[4] = fit_params[4] - self.halfCropRange + sepSpotDict["xcpeak"]
                    fit_params[5] = fit_params[5] - self.halfCropRange + sepSpotDict["ycpeak"]

                    spotDetailDict = {}
                    spotDetailDict["fullFittingParam"] = fit_params
                    spotDetailDict["A"] = fit_params[0]
                    spotDetailDict["B"] = fit_params[1]
                    spotDetailDict["C"] = fit_params[2]
                    spotDetailDict["Am"] = fit_params[3]
                    spotDetailDict["xCenter"] = fit_params[4]
                    spotDetailDict["yCenter"] = fit_params[5]
                    spotDetailDict["sigma_x"] = fit_params[6]
                    spotDetailDict["sigma_y"] = fit_params[7]
                    spotDetailDict["theta"] = fit_params[8]

                    # fitParamsFrameDict[str(spotID)] = fit_params
                    fitParamsFrameDict[str(spotID)] = spotDetailDict

                    fitUncertDict[str(spotID)] = uncert_cov

                fitParamsFrameDict["filePath"] = frameDict["filePath"]
                fitParamsFrameDict["numberOfSpot"] = numberOfSpot
                fitParamsFrameDict["FittingTime"] = SPAFrameTimer.tocvalue()

                return fitParamsFrameDict, fitUncertDict

        print("SPAMode")

        if sepDict == None:
            if self.sepComplete == False:
                print("Runing SEPMode to get Rough range")
                self.sepMode()
        else:
            self.sepDict = sepDict
            self.createNGaussDict()

        SPATimer.tic()

        self.SPAResultRawDict = {}
        self.SPAUncertDict = {}
        for frameID, frameSEPDict in self.sepDict.items():
            self.SPAResultRawDict[str(frameID)], self.SPAUncertDict[str(frameID)] = applySPA(frameID, frameSEPDict)

        print("save to :" + self.SPACSVNameRaw)

        self.integrateFittedPeakIntensity()

        for i in range(self.csvHeaderLength):
            self.SPACSVHeader += self.SPAparameterHeader

        self.saveToCSV([self.SPACSVHeader], self.SPACSVNameRaw)
        self.saveToCSV(self.convertSPADictIntoCSVWriteArray(self.SPAResultRawDict), self.SPACSVNameRaw)
        self.saveDictToPLK(self.SPAResultRawDict,
                           self.timeStamp + "_" + self.configDict["csvRemark"] + "_RawSPADict")
        self.saveDictToPLK(self.SPAUncertDict,
                           self.timeStamp + "_" + self.configDict["csvRemark"] + "_RawSPAUncertDict")

        print("SPA Complete")
        SPATimer.toc()

        return self.SPAResultRawDict

    def integrateFittedPeakIntensity(self, spaDict=""):
        inTimer = TicToc()
        inTimer.tic()

        if spaDict == "":
            spaDict = self.SPAResultRawDict

        for frameID, frameDict in spaDict.items():
            numberOfSpot = frameDict["numberOfSpot"]
            totalIntensity = 0

            for spotID in range(int(numberOfSpot)):
                spotDict = frameDict[str(spotID)]

                Am = spotDict["Am"]
                xCenter = spotDict["xCenter"]
                yCenter = spotDict["yCenter"]
                sigma_x = spotDict["sigma_x"]
                sigma_y = spotDict["sigma_y"]
                theta = spotDict["theta"]

                xUpperLimit = xCenter + self.halfCropRange
                xLowerLimit = xCenter - self.halfCropRange

                yUpperLimit = yCenter + self.halfCropRange
                yLowerLimit = yCenter - self.halfCropRange

                spotDict["integratedIntensity"], spotDict["integratedIntensityError"] = dblquad(
                    lambda x, y: fitFunc.gauss2D(x, y, Am, xCenter, yCenter, sigma_x, sigma_y, theta), yLowerLimit,
                    yUpperLimit, lambda x: xLowerLimit, lambda x: xUpperLimit)

                totalIntensity += spotDict["integratedIntensity"]
            frameDict["totalIntensity"] = totalIntensity

            if int(frameID) % 50 == 0:
                print("Integrating Frame ID:", frameID, end=', ')
                inTimer.toc()

        for frameID, frameDict in spaDict.items():
            numberOfSpot = frameDict["numberOfSpot"]
            for spotID in range(int(numberOfSpot)):
                spotDict = frameDict[str(spotID)]
                spotDict["integratedIntensityRatio"] = spotDict["integratedIntensity"] / frameDict["totalIntensity"]

        self.SPAResultRawDict = spaDict

        return self.SPAResultRawDict

    def ellipticalCorrection(self):

        def gatherXYCenterFromSPADict():
            gatheredXCenterCoorList = []
            gatheredYCenterCoorList = []

            for frameID, frameDict in self.SPAResultRawDict.items():
                numberOfSpot = frameDict["numberOfSpot"]
                for spotID in range(int(numberOfSpot)):
                    gatheredXCenterCoorList.append(frameDict[str(spotID)]["xCenter"])
                    gatheredYCenterCoorList.append(frameDict[str(spotID)]["yCenter"])

            return np.array(gatheredXCenterCoorList), np.array(gatheredYCenterCoorList)

        def fitEllipse():  # x, y
            x, y = gatherXYCenterFromSPADict()
            x = x[:, np.newaxis]
            y = y[:, np.newaxis]
            D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
            S = np.dot(D.T, D)
            C = np.zeros([6, 6])
            C[0, 2] = C[2, 0] = 2
            C[1, 1] = -1
            E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
            n = np.argmax(np.abs(E))
            a = V[:, n]
            return a

        def cart2pol(x, y):
            rho = np.sqrt(x ** 2 + y ** 2)
            phi = np.rad2deg(np.arctan2(y, x))
            return (rho, phi)

        self.SPAResultEllipticalCorrectedDict = self.SPAResultRawDict.copy()

        a = fitEllipse()
        aa = np.zeros_like(a)

        th = 0.5 * np.arctan(a[2 - 1] / (a[1 - 1] - a[3 - 1]))
        aa[1 - 1] = a[1 - 1] * np.cos(th) * np.cos(th) + a[2 - 1] * np.sin(th) * np.cos(th) + a[3 - 1] * np.sin(
            th) * np.sin(th)
        aa[2 - 1] = 0
        aa[3 - 1] = a[1 - 1] * np.sin(th) * np.sin(th) - a[2 - 1] * np.sin(th) * np.cos(th) + a[3 - 1] * np.cos(
            th) * np.cos(th)
        aa[4 - 1] = a[4 - 1] * np.cos(th) + a[5 - 1] * np.sin(th)
        aa[5 - 1] = -a[4 - 1] * np.sin(th) + a[5 - 1] * np.cos(th)
        aa[6 - 1] = a[6 - 1]

        X0 = -aa[4 - 1] / 2 / aa[1 - 1]
        Y0 = -aa[5 - 1] / 2 / aa[3 - 1]
        x0 = X0 * np.cos(th) - Y0 * np.sin(th)
        y0 = X0 * np.sin(th) + Y0 * np.cos(th)

        A = np.sqrt((aa[1 - 1] * X0 ** 2 + aa[3 - 1] * Y0 ** 2 - aa[6 - 1]) / aa[3 - 1])
        B = np.sqrt((aa[1 - 1] * X0 ** 2 + aa[3 - 1] * Y0 ** 2 - aa[6 - 1]) / aa[1 - 1])

        if B > A:
            A, B = B, A
            th = th + np.pi / 2

        for frameID, frameDict in self.SPAResultEllipticalCorrectedDict.items():
            numberOfSpot = frameDict["numberOfSpot"]
            for spotID in range(int(numberOfSpot)):
                spotDict = frameDict[str(spotID)]

                x = spotDict["xCenter"]
                y = spotDict["yCenter"]

                xx = x - x0
                yy = y - y0
                XX = xx * np.cos(-th) - yy * np.sin(-th)
                YY = xx * np.sin(-th) + yy * np.cos(-th)
                XX = XX * A / B
                xx = XX * np.cos(th) - YY * np.sin(th)
                yy = XX * np.sin(th) + YY * np.cos(th)
                spotDict["xCenter"] = xx
                spotDict["yCenter"] = yy

                spotDict["polarCorr"] = cart2pol(xx, yy)
                spotDict["k"] = spotDict["polarCorr"][0]
                spotDict["thetaPolarCorr"] = spotDict["polarCorr"][1]

                frameDict[str(spotID)] = spotDict
            self.SPAResultEllipticalCorrectedDict[str(frameID)] = frameDict

        self.saveToCSV([self.SPACSVHeader], self.SPACSVNameEllipticalCorrected)
        self.saveToCSV(self.convertSPADictIntoCSVWriteArray(self.SPAResultEllipticalCorrectedDict),
                       self.SPACSVNameEllipticalCorrected)
        self.saveDictToPLK(self.SPAResultEllipticalCorrectedDict,
                           self.timeStamp + "_" + self.configDict["csvRemark"] + "_EllipticalCorrectedSPADict")

        return self.SPAResultEllipticalCorrectedDict


class utility:
    def __init__(self, SPAdict, zeroAngularPosition):
        self.SPAdict = SPAdict

        self.thetaArray = np.array(self.gatherItemFromDict("thetaPolarCorr", returnFramewise=True))
        self.adjThetaArray = self.adjSpotAngle(zeroAngularPosition)
        self.ampRatioArray = np.array(self.gatherItemFromDict("integratedIntensityRatio",
                                                              returnFramewise=True))
        self.ampRatioArray, self.adjThetaArray = self.clusterDomain(self.adjThetaArray, self.ampRatioArray)

        self.makeColorMap()

    def makeColorMap(self):

        nbins = 10000
        redAlphaDict = {'red': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)), 'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
                        'blue': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))}
        greenAlphaDict = {'red': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)), 'green': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
                          'blue': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))}
        blueAlphaDict = {'red': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)), 'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
                         'blue': ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0))}
        self.redAlpha = LinearSegmentedColormap('redAlpha', redAlphaDict, N=nbins)
        self.greenAlpha = LinearSegmentedColormap('greenAlpha', greenAlphaDict, N=nbins)
        self.blueAlpha = LinearSegmentedColormap('blueAlpha', blueAlphaDict, N=nbins)

    def gatherItemFromDict(self, searchKey, returnFramewise=False):
        dataDict = self.SPAdict
        returnList = []

        if returnFramewise:
            for frame in range(len(dataDict)):
                frameList = []
                for spotID in range(int(dataDict[str(frame)]["numberOfSpot"])):
                    frameList.append(dataDict[str(frame)][str(spotID)][searchKey])
                returnList.append(np.array(frameList))

        else:
            for frame in range(len(dataDict)):
                for spotID in range(int(dataDict[str(frame)]["numberOfSpot"])):
                    returnList.append(dataDict[str(frame)][str(spotID)][searchKey])
        return np.array(returnList)

    def adjSpotAngle(self, firstSpotMean, threshold=200):
        def adjAngle(inputAngle):
            if inputAngle < 0:
                inputAngle += 360
            if 330 < inputAngle:
                inputAngle -= 360
            elif inputAngle < 30:
                inputAngle -= 0
            elif 30 < inputAngle < 90:
                inputAngle -= 60
            elif 90 < inputAngle < 150:
                inputAngle -= 120
            elif 150 < inputAngle < 210:
                inputAngle -= 180
            elif 210 < inputAngle < 270:
                inputAngle -= 240
            elif 270 < inputAngle < 330:
                inputAngle -= 300
            return inputAngle

        if type(self.thetaArray) is float:
            self.thetaArray -= firstSpotMean
            self.thetaArray = adjAngle(self.thetaArray)
            return self.thetaArray
        else:
            array = np.copy(self.thetaArray)
            array -= firstSpotMean
            for i in range(len(array)):
                for j in range(len(array[i])):
                    array[i][j] = adjAngle(array[i][j])

            for i in range(len(array)):
                array[i] = array[i][array[i] < threshold]
            return array

    def clusterDomain(self, adjedThetaList, ampRatioList, clusterDegWindow=2):
        def cluster(items, key_func):
            items = sorted(items)
            clustersList = [[items[0]]]
            for item in items[1:]:
                cluster = clustersList[-1]
                last_item = cluster[-1]
                if key_func(item, last_item):
                    cluster.append(item)
                else:
                    clustersList.append([item])
            return clustersList

        returnDomainThetaList = []
        returnDomainAmpList = []
        for thetaInFrameList, ampInFrameList in zip(adjedThetaList, ampRatioList):
            domainAngleList = []
            domainAmpList = []
            clusterListInFrame = cluster(sorted(thetaInFrameList.tolist()),
                                         lambda curr, prev: curr - prev < clusterDegWindow)
            for domain in clusterListInFrame:
                domainAngle = np.mean(domain)
                domainAmp = 0
                for angle in domain:
                    angleIndexInthetaInFrameList = list(thetaInFrameList).index(angle)
                    domainAmp += ampInFrameList[angleIndexInthetaInFrameList]
                domainAngleList.append(domainAngle)
                domainAmpList.append(domainAmp)

            returnDomainThetaList.append(domainAngleList)
            returnDomainAmpList.append(domainAmpList)
        return np.array(returnDomainAmpList), np.array(returnDomainThetaList)

    def selectTheatRange(self, rList, thetaList, thetaMin, thetaMax, returnRad=True):
        rList = np.concatenate(rList)
        thetaList = np.concatenate(thetaList)
        returnRList = []
        returnThetaList = []

        for i in range(len(thetaList)):
            if thetaMin < thetaList[i] < thetaMax:
                returnRList.append(rList[i])
                returnThetaList.append(thetaList[i])
        if returnRad:
            return returnRList, np.radians(returnThetaList)
        else:
            return returnRList, returnThetaList