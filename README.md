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
- Linux and Mac (Preferable for easy sep install)
- Windows

# Install
Download the package from github

or

Run (if git is installed):
```
git clone https://github.com/klwlau/LEED-SEPSPA
```

# Config SEPSAP 
To run SEPSAP there are some parameter that need to be tune before use, all parameters are stored in `configList.json`.

Parameter | Type | Description
------------ | -------------| -------------
dataFolderName | string | path to data (eg. `"C:/data"`)
CSVwriteBuffer | int | how often sepspa save result into csv 
csvRemark | string |add remark to csv output file name
sepSingleCoreDebugMode | bool| disable multicore runtime when set to `false`
|||
maskConfig| dict| create a ring mask over the analyzed data
mask_x_center|int| mask x center location
mask_y_center|int| mask y center location
innerRadius|int| ring mask inner radius
outerRadius|int| ring mask outer radius
|||
SEPParameters| dict| parameters that control sep





