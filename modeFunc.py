
def fittingMode():
    # init first row in CSV file
    writeBufferArray = [["FileID", "File Name", "Number of Spots",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C",
                         "Am", "x_0", "y_0", "sigma_x", "sigma_y", "shape_x", "shape_y", "theta", "A", "B", "C"]]
    counter = 0
    fileAmount = len(fileList)

    for fileName in fileList:
        # need to add all parameters back
        templist, numberOfSpots = findSpot(fileName, configList["findSpotParameters"]["searchThreshold"],
                                           mask, scaleDownFactor=configList["findSpotParameters"]["scaleDownFactor"],
                                           showSpots=False, fileID=counter, saveMode=configList["saveMode"])
        writeBufferArray.append(templist)

        print(counter, ",", numberOfSpots, ",", fileName, ",", counter / fileAmount * 100, "%")

        if counter % CSVwriteBuffer == 0:
            saveToCSV(writeBufferArray, CSVName)
            writeBufferArray = []
            print("---------------save to CSV---------------")

        if counter == (fileAmount - 1):
            saveToCSV(writeBufferArray, CSVName)
            print("---------------save to CSV---------------")
        counter += 1

def sepMode():
    # init first row in CSV file
    writeBufferArray = [["FileID", "File Name", "Number of Spots",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta",
                         "Am", "x", "y", "xpeak", "ypeak", "a", "b", "theta"]]
    counter = 0
    fileAmount = len(fileList)

    for fileName in fileList:
        # need to add all parameters back
        templist, numberOfSpots = findSpot(fileName, configList["findSpotParameters"]["searchThreshold"],
                                           mask, scaleDownFactor=configList["findSpotParameters"]["scaleDownFactor"],
                                           showSpots=False, fileID=counter, saveMode=configList["saveMode"],fittingMode=False)
        writeBufferArray.append(templist)

        print(counter, ",", numberOfSpots, ",", fileName, ",", counter / fileAmount * 100, "%")

        if counter % CSVwriteBuffer == 0:
            saveToCSV(writeBufferArray, CSVName)
            writeBufferArray = []
            print("---------------save to CSV---------------")

        if counter == (fileAmount - 1):
            saveToCSV(writeBufferArray, CSVName)
            print("---------------save to CSV---------------")
        counter += 1



def testMode():
    # need to add testMode parameters
    print("File name: ", fileList[configList["testModeParameters"]["testModeFileID"]])
    findSpot(fileList[configList["testModeParameters"]["testModeFileID"]],
             configList["findSpotParameters"]["searchThreshold"], mask,
             scaleDownFactor=configList["testModeParameters"]["scaleDownFactor"],
             plotSensitivity_low=configList["testModeParameters"]["plotSensitivity_low"],
             plotSensitivity_up=configList["testModeParameters"]["plotSensitivity_up"],
             showSpots=configList["testModeParameters"]["showSpots"],
             plotFittedFunc=configList["testModeParameters"]["plotFittedFunc"],
             printParameters=configList["testModeParameters"]["printParameters"])