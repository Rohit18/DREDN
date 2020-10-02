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
import cv2

directory = os.getcwd() + '\\LR_512\\'
pathlist = Path(directory).iterdir()


for path in pathlist:

    print(path)
    
    img = np.array(scipy.ndimage.imread(str(path)))
    img = cv2.resize(img, dsize=(64, 64), interpolation = cv2.INTER_CUBIC)
    filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".")]
    skimage.io.imsave(os.getcwd() + '\\LR_64\\' + filename + '.png', img)