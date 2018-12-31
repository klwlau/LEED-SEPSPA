class fitting:


    def __init__(self,configFilePath="configList.json"):
        import time
        import glob
        self.start_time = time.time()
        print("Program Started, Loading Libraries")
        import datetime
        print("Start Loading UsedFunc")
        from scipy.optimize import curve_fit
        import sep
        from PIL import Image
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from pytictoc import TicToc
        import csv, itertools, json, os, shutil, ntpath

        self.configFilePath = configFilePath
        self.configList = json.load(open(self.configFilePath))
        self.globalCounter = 0
        self.timeStamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        self.dataFolderName = self.configList["dataFolderName"]

    def makeResultDir(self):
        '''make a new directory storing fitting result if it does not exists'''
        if not os.path.exists(os.path.join(os.curdir, "Result")):
            os.makedirs(os.path.join(os.curdir, "Result"))
            print("make Result Dir")

    def initlizing(self):
        self.makeResultDir()
        self.CSVName = "./Result/" + self.timeStamp + "_" + self.configList["csvNameRemark"] + ".csv"


    def printSaveStatus(self):
        if self.globalCounter != 0:
            elapsedTime = ((time.time() - self.start_time) / 60)
            totalTime = elapsedTime / (self.globalCounter / fileAmount)
            timeLeft = totalTime - elapsedTime

            print("---Elapsed Time: %.2f / %.2f Minutes ---" % (elapsedTime, totalTime)
                  + "---Time Left: %.2f  Minutes ---" % timeLeft
                  + "--save to" + self.CSVName)



    def printConfigList(self):
        for i in self.configList:
            print (i)

    def testMode(self):
        print("TestMode")

    def SEPMode(self):
        print("SEPMode")

    def SPAMode(self):
        print("SPAMode")