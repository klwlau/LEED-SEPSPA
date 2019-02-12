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
from scipy.integrate import dblquad
import sepspa.fitFunc as fitFunc


class fitting:

    def __init__(self, configFilePath="configList.json", listLength="Full"):
        np.set_printoptions(precision=3, suppress=True)

        self.start_time = time.time()
        self.configFilePath = configFilePath
        self.configList = json.load(open(self.configFilePath))
        self.timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        # loading confingList
        self.dataFolderName = self.configList["dataFolderName"]
        self.halfCropRange = self.configList["SEPParameters"]["cropRange"] // 2
        self.searchThreshold = self.configList["SEPParameters"]["searchThreshold"]
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
        if listLength != "Full":
            self.fileList = self.fileList[:listLength]
        self.CSVwriteBuffer = self.configList["CSVwriteBuffer"]
        self.preStart()
        self.csvHeaderLength = 15
        self.fittingBoundDict = {}
        self.fittingIntDict = {}
        self.multipleSpotInFrameThreshold = self.configList["SPAParameters"]["multipleSpotInFrameRange"] / 2

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

    def genIntCondittion(self, spotID, frameID, sepSpotDict, numOfGauss=1):
        intGuess = self.configList["SPAParameters"]["backgroundIntGuess"].copy()

        for i in range(numOfGauss):
            if i == 0:
                intGuess += [sepSpotDict["Am"]]
                intGuess += [self.halfCropRange]
                intGuess += [self.halfCropRange]
                intGuess += [sepSpotDict["a"]]
                intGuess += [sepSpotDict["b"]]
                intGuess += [sepSpotDict["theta"]]
            else:
                if self.configList["SPAParameters"]["smartConfig"]:
                    tempMinorGaussianIntGuess = self.configList["SPAParameters"]["minorGaussianIntGuess"].copy()
                    tempMinorGaussianIntGuess[2] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][0]
                    tempMinorGaussianIntGuess[1] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][1]
                    intGuess += tempMinorGaussianIntGuess
                else:
                    intGuess += self.configList["SPAParameters"]["minorGaussianIntGuess"]

        return intGuess

    def genFittingBound(self, spotID, frameID, numOfGauss=1):
        guessUpBound = self.configList["SPAParameters"]["backgroundGuessUpperBound"].copy()
        guessLowBound = self.configList["SPAParameters"]["backgroundGuessLowerBound"].copy()

        for i in range(numOfGauss):
            tempSpotUpBound = self.configList["SPAParameters"]["gaussianUpperBoundTemplate"].copy()
            tempSpotLowBound = self.configList["SPAParameters"]["gaussianLowerBoundTemplate"].copy()
            if i == 0:
                tempSpotUpBound[2] = self.halfCropRange + self.configList["SPAParameters"]["majorGaussianXYRange"]
                tempSpotUpBound[1] = self.halfCropRange + self.configList["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[2] = self.halfCropRange - self.configList["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[1] = self.halfCropRange - self.configList["SPAParameters"]["majorGaussianXYRange"]
            else:
                tempSpotUpBound[2] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][0] + \
                                     self.configList["SPAParameters"]["majorGaussianXYRange"]
                tempSpotUpBound[1] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][1] + \
                                     self.configList["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[2] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][0] - \
                                      self.configList["SPAParameters"]["majorGaussianXYRange"]
                tempSpotLowBound[1] = self.neighborSpotDict[str(frameID)][str(spotID)][i - 1][1] - \
                                      self.configList["SPAParameters"]["majorGaussianXYRange"]
            guessUpBound += tempSpotUpBound
            guessLowBound += tempSpotLowBound
        return [guessLowBound, guessUpBound]

    def calRSquareError(self, fittedArray, rawArray):
        """calculate R Square error"""
        errorSquare = (fittedArray - rawArray) ** 2
        SS_tot = np.sum((fittedArray - np.mean(fittedArray)) ** 2)
        SS_res = np.sum(errorSquare)
        return (SS_res / SS_tot)

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
        # self.maxSpotInFrame = max(frameDict["numberOfSpot"], self.maxSpotInFrame)
        # self.sepDict["maxSpotInFrame"] = self.maxSpotInFrame

    def testMode(self):
        print("TestMode")

    def sepMode(self):

        def parallelSEP(fileID, filePath):
            imageArray = self.readLEEDImage(filePath)
            imageArray = self.compressImage(imageArray)
            # imageArray = self.applyMask(imageArray)

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
        sepCSVHeader = ["FileID", "File Path", "Number of Spots"]
        SEPparameterHeader = ["Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta"]

        for i in range(self.csvHeaderLength):
            sepCSVHeader += SEPparameterHeader

        self.saveToCSV([sepCSVHeader], self.SEPCSVName)

        if self.configList["sepSingleCoreDebugMode"] != True:
            with Parallel(n_jobs=-1, verbose=2) as parallel:
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
        test = self.halfCropRange * 2
        # xi, yi = np.mgrid[0:self.halfcropRange * 2, 0:self.halfcropRange * 2]
        xi, yi = np.mgrid[0:test, 0:test]

        xyi = np.vstack([xi.ravel(), yi.ravel()])
        numOfGauss = int((len(fit_params) - 3) / len(self.configList["SPAParameters"]["gaussianUpperBoundTemplate"]))

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
                                                           + self.configList["SPAParameters"][
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

    def SPAMode(self):

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
                    if self.configList["SPAParameters"]["adaptiveGaussianFitting"]:
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

                    if self.configList["SPAParameters"]["saveFitFuncPlot"]:
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

        def convertSPADictIntoCSVWriteArray(SPADict):
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
                    spotArray.append(frameDict[str(spotID)]["fullFittingParam"][3:9])
                    spotArray.append(frameDict[str(spotID)]["fullFittingParam"][0:3])

                spotArray = list(itertools.chain.from_iterable(spotArray))
                frameWriteArray += spotArray

                CSVWriteArray.append(frameWriteArray)

            return CSVWriteArray

        print("SPAMode")
        if self.sepComplete == False:
            print("Runing SEPMode to get Rough range")
            self.sepMode()

        SPATimer.tic()

        self.SPAResultDict = {}
        self.SPAUncertDict = {}
        for frameID, frameSEPDict in self.sepDict.items():
            self.SPAResultDict[str(frameID)], self.SPAUncertDict[str(frameID)] = applySPA(frameID, frameSEPDict)

        print("save to :" + self.SPACSVName)

        SPACSVHeader = ["FileID", "File Path", "Number of Spots", "Fitting Time"]
        SPAparameterHeader = ["Am", "x", "y", "sigma_x", "sigma_y", "theta", "A", "B", "Constant"]

        for i in range(self.csvHeaderLength):
            SPACSVHeader += SPAparameterHeader

        self.saveToCSV([SPACSVHeader], self.SPACSVName)
        self.saveToCSV(convertSPADictIntoCSVWriteArray(self.SPAResultDict), self.SPACSVName)
        self.saveDictToPLK(self.SPAResultDict, self.timeStamp + "_" + self.configList["csvNameRemark"] + "_SPADict")
        self.saveDictToPLK(self.SPAUncertDict,
                           self.timeStamp + "_" + self.configList["csvNameRemark"] + "_SPAUncertDict")

        print("SPA Complete")
        SPATimer.toc()

        return self.SPAResultDict

    def integrateFittedPeakIntensity(self, spaDict=""):
        if spaDict == "":
            spaDict = self.SPAResultDict

        for frameID, frameDict in spaDict.items():
            numberOfSpot = frameDict["numberOfSpot"]

            for spotID in range(int(numberOfSpot)):
                spotDict = frameDict[str(spotID)]

                Am = spotDict["Am"]
                xCenter = spotDict["xCenter"]
                yCenter = spotDict["yCenter"]
                sigma_x = spotDict["sigma_x"]
                sigma_y = spotDict["sigma_y"]
                theta = spotDict["theta"]

                upperLimit = xCenter +self.halfCropRange
                lowerLimit = xCenter -self.halfCropRange


                spotDict["integrateIntensity"] = dblquad(fitFunc.gauss2D(x,y,Am,xCenter,yCenter,sigma_x,sigma_y,theta),
                                                         lowerLimit,upperLimit,lowerLimit,upperLimit)
                print(spotDict["integrateIntensity"])













        self.SPAResultDict = spaDict

        return self.SPAResultDict



























