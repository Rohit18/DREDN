import os
import tifffile as tiff
import numpy as np
import slidingwindow as sw
from pathlib import Path
from matplotlib import pyplot as plt
import scipy
from PIL import Image
import skimage
import gdal




""" directory = os.getcwd() + '\\MOD09GA_SWIR1\\'
pathlist = Path(directory).iterdir()

#print(directory)

for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))

    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, 128, 0)

    for i, window in enumerate(windows):
        subset = img[ window.indices() ]
        filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
        subset = skimage.img_as_int(subset)
        skimage.io.imsave('C:\\Users\\rohit\\Workspace\\Python\\MODIS\\SWIR1_128\\' + filename + '_' + str(i) + '.tif', subset)


directory = os.getcwd() + '\\MOD09GA_Green\\'
pathlist = Path(directory).iterdir()


for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))

    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, 128, 0)

    for i, window in enumerate(windows):
        subset = img[ window.indices() ]
        filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
        subset = skimage.img_as_int(subset)
        skimage.io.imsave('C:\\Users\\rohit\\Workspace\\Python\\MODIS\\Green_128\\' + filename + '_' + str(i) + '.tif', subset)



directory = os.getcwd() + '\\MOD09GA_SWIR2\\'
pathlist = Path(directory).iterdir()

#print(directory)

for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))

    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, 128, 0)

    for i, window in enumerate(windows):
        subset = img[ window.indices() ]
        filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
        subset = skimage.img_as_int(subset)
        skimage.io.imsave('C:\\Users\\rohit\\Workspace\\Python\\MODIS\\SWIR2_128\\' + filename + '_' + str(i) + '.tif', subset)


directory = os.getcwd() + '\\MOD09GA_Blue\\'
pathlist = Path(directory).iterdir()

#print(directory)

for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))

    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, 128, 0)

    for i, window in enumerate(windows):
        subset = img[ window.indices() ]
        filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
        subset = skimage.img_as_int(subset)
        skimage.io.imsave('C:\\Users\\rohit\\Workspace\\Python\\MODIS\\Blue_128\\' + filename + '_' + str(i) + '.tif', subset)


directory = os.getcwd() + '\\MOD09GA_Red\\'
pathlist = Path(directory).iterdir()

#print(directory)

for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))

    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, 128, 0)

    for i, window in enumerate(windows):
        subset = img[ window.indices() ]
        filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
        subset = skimage.img_as_int(subset)
        skimage.io.imsave('C:\\Users\\rohit\\Workspace\\Python\\MODIS\\Red_128\\' + filename + '_' + str(i) + '.tif', subset) """



#directory = os.getcwd() + '\\MOD09GA_NIR\\'
directory = os.getcwd() + '\\M31\\'
pathlist = Path(directory).iterdir()

#print(directory)

for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))

    windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, 256, 0.3)

    for i, window in enumerate(windows):
        subset = img[ window.indices() ]
        filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
        skimage.io.imsave(os.getcwd() + '\\M31_256\\' + filename + '_' + str(i) + '.png', subset) 