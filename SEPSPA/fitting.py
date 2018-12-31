class fitting:
    from scipy.optimize import curve_fit
    import sep
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from pytictoc import TicToc
    import csv, itertools, json, os, shutil, ntpath

    def __int__(self,configFilePath="configList.json"):
        self.configFilePath = configFilePath


    def testMode(self):
        print("TestMode")

    def SEPMode(self):
        print("SEPMode")

    def SPAMode(self):
        print("SPAMode")