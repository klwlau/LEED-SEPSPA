class fitting:


    def __init__(self,configFilePath="configList.json"):
        import time
        import glob
        start_time = time.time()
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


    def printConfigList(self):
        for i in self.configList:
            print (i)

    def testMode(self):
        print("TestMode")

    def SEPMode(self):
        print("SEPMode")

    def SPAMode(self):
        print("SPAMode")