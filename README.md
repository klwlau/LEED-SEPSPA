# LEED-SEPSPA

A tools to locate and analyse LEED spot

# Package Requirements

- Python 3.6+
- sep 1.0.3+ (https://github.com/kbarbary/sep)
- joblib 0.13.2+
- numba 0.41.0+
- scipy 1.3.1+
- numpy 1.16.4+
- matplotlib 3.2.0+
- pytictoc 1.5.0+

# Support platform

- Linux and Mac (Preferable for easy sep package install)
- Windows

# Install

Download the package from github

or

Run (if git is installed):
```
git clone https://github.com/klwlau/LEED-SEPSPA
```

# Example code

To run sepspa **test mode** with a config file:

```python
from sepspa.fitting import fitting

example = fitting(configFilePath="configList.json")
example.testMode()
```

To run sepspa **sepspa mode** with a config file:

```python
from sepspa.fitting import fitting

example = fitting(configFilePath="configList.json")
example.sepspaMode()
```

To run sepspa **sep mode** with a config file:

```python
from sepspa.fitting import fitting

example = fitting(configFilePath="configList.json")
example.sepMode()
```

# Result output
|File name | Type | Location| Description|
|------------: | :-------------:| :------------:| :-------------|
|`<time_stamp>.json`|`.json`|`./Log/`| mirror `config.json` used in last run|
|`<time_stamp>_SPARaw.csv`|`.csv`|`./Result/`| result output from `spa`|
|`<time_stamp>_SEP.csv`|`.csv`|`./Result/`| result output from `sep`|
|`<time_stamp>__RawSPADict.pkl`|`.pkl`|`<data folder>/pythonObj/`|a python dictionary object that stored `spa` output|
|`<time_stamp>__RawSPAUncertDict.pkl`|`.pkl`|`<data folder>/pythonObj/`|a python dictionary object that stored `spa` error|
|`<time_stamp>__SEPDict.pkl`|`.pkl`|`<data folder>/pythonObj/`|a python dictionary object that stored `sep` output|
|`<data file name>_<spot number>.png`|`.png`|`<data folder>/fitFuncFig_<user remark>/`| figure output if `saveFitFuncPlot` enable
|`<data file name>_SEP.png`|`.png`|`<data folder>/SEPResult/`| figure output if `saveSEPResult` enable

# Config SEPSAP 

To run SEPSAP there are some parameter that need to be tune before use, all parameters are stored in `configList.json`.

|Parameter | Type | Description|
|------------: | :-------------:| :-------------|
|`dataFolderName`|`string` | path to data (eg. `"C:/data"`)|
|`CSVwriteBuffer`|`int` | how often sepspa save result into csv| 
|`csvRemark`|`string` |add remark to csv output file name|
|`sepSingleCoreDebugMode`|`bool`| disable multicore runtime when set to `false`|
|------------ | ------| ---------------------------------------|
|`maskConfig`|`dict`| create a ring mask over the analyzed data|
|`mask_x_center`|`int`| mask x center location|
|`mask_y_center`|`int`| mask y center location|
|`innerRadius`|`int`| ring mask inner radius|
|`outerRadius`|`int`| ring mask outer radius|
|------------ | ------| ---------------------------------------|
|`SEPParameters`|`dict`| parameters that control `sep`|
|`searchThreshold`|`int`| control the sensitivity of `sep`. When the value increase, `sep` will ignore low intensity spot|  
|`cropRange`|`int`| a N by N crop around the identified spot by `sep` that will pass into `spa`| 
|`scaleDownFactor`|`float`| compress pixel intensity to prevent `sep` overflow by `outputPixel = inputPixel // scaleDownFactor`. Use only when pixel intensity is too high. This calculation only use within `sep`. It does not apply to `spa`.|
|`saveSEPResult`|`bool`| save individual `sep` result. (slow down runtime performance)|
|------------ | ------| ---------------------------------------|
|`testModeParameters` |`dict`| parameters that control testmode (only run `sepspa` on one file)|
|`showSpots`|`bool`|plot `sep` result if set to `true`|
|`printSpotRoughRangeArray`|`bool`|print out `sep` result in terminal or command prompt if set to `true`|
|`fittingMode`|`bool`| run `spa` mode if set to `true`|
|`testModeFileID`|`int`| selected a file from the data folder.| 
|`plotSensitivity`|`int`| sensitivity of colour scale used in plotting|
|`sepPlotColourMin`|`int`| `sep` mode result plot colour scale minimum|
|`sepPlotColourMax`|`int`|`sep` mode result plot colour scale maximum| 
|`scaleDownFactor`|`int`| compress pixel intensity to prevent `sep` overflow by `outputPixel = inputPixel // scaleDownFactor`. Use only when pixel intensity is too high. This calculation only use within `sep`. It does not apply to `spa`.|
|`printFittedParameters`|`bool`|print `spa` result if set to `true`|
|`plotFittedFunc`|`bool`|plot `spa` result if set to `true`|
|------------ | ------| ---------------------------------------|
|`SPAParameters`|`dict`|parameters that control `spa`|
|`backgroundGuessUpperBound`|`array`|the upper bond of background fitting `[a,b,c]` in fitting function `ax+by+c`| 
|`backgroundGuessLowerBound`|`array`|the lower bond of background fitting `[a,b,c]` in fitting function `ax+by+c`|
|`adaptiveGaussianFitting`|`bool`| search within a N by N box set in `multipleSpotInFrameRange` to see if there are other in identified spot. if set `true`, add another 2D gaussian to improve fitting.|
|`smartConfig`|`bool`| use sep result for adding 2D gaussian in `adaptiveGaussianFitting`|
|`multipleSpotInFrameRange`|`int`| search within a N by N box to see if there are other in identified spot. when adaptiveGaussianFitting is `true`, add another 2D gaussian to improve fitting.|
|`majorGaussianXYRange`|`float`| A n by n box limiting center of 2D gaussian|
|`gaussianUpperBoundTemplate`|`array`|the upper bond of gaussian function, `[Amp, x_0, y_0, sigma_x, sigma_y, theta]`| 
|`gaussianLowerBoundTemplate`|`array`| the lower bond of gaussian function, `[Amp, x_0, y_0, sigma_x, sigma_y, theta]`|
|`backgroundIntGuess`|`array`| the initial guess of background `[a,b,c]` in fitting function `ax+by+c`|
|`minorGaussianIntGuess`|`array`| 2D gaussian initial guess use in `adaptiveGaussianFitting`|
|`saveFitFuncPlot`|`bool`| save a plot for every fitted plot, if set true|
|`saveFitFuncPlotFileRemark`|`string`| add remark to fitted plot output file name|

