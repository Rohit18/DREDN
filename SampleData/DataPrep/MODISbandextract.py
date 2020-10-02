import numpy as np
import h5py
from pyhdf.SD import SD, SDC
import gdal
import os
import matplotlib.pyplot as plt
from pathlib import Path

# f = h5py.File('MYD09GA.A2019116.h10v05.006.2019118025813.hdf', 'r')
# ls = list(f.keys())
# f.close()

# print(ls)

def updateGeoTransforms(srcGeo, xOff, yOff):
    """
    Create a new geotransform list based on a source geoTransform and an offset
    Returns a list of 6
    :param srcGeo: The geoTransfor of the uncroped dataset retrieved using GetGeoTransform()
    :param xOff: x offset used for cropping
    :param yOff: y offset used for cropping
    """
    out = [srcGeo[0] + xOff * srcGeo[1], srcGeo[1], srcGeo[2], srcGeo[3] + yOff * srcGeo[5], srcGeo[4], srcGeo[5]]
    return out



directory = os.getcwd() + '\\MQ\\'
pathlist = Path(directory).iterdir()

for path in pathlist:


    #name = 'MYD09GA.A2019116.h10v05.006.2019118025813.hdf'
    #sds = gdal.Open(name, gdal.GA_ReadOnly).GetSubDatasets()

    sds = gdal.Open(str(path), gdal.GA_ReadOnly).GetSubDatasets()

    #print(sds[1])

    vi = gdal.Open(sds[1][0])
    vi_np = vi.ReadAsArray()
    #print(vi_np.shape)

    #plt.imshow(vi_np)
    #plt.show()

    geoT = vi.GetGeoTransform()
    proj = vi.GetProjection()

    filename = str(os.path.basename(path))[:str(os.path.basename(path)).find(".hdf")] + '_B1.tif'
    ##outfile_name = 'new.tif'


    driver = gdal.GetDriverByName('GTiff')

    width = 4800
    height = 4800
    xOff = 0
    yOff = 0

    #dataset = driver.Create(filename, width, height, 1, gdal.GDT_Int16)
    dataset = driver.Create(filename, width, height, 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform(updateGeoTransforms(geoT, xOff, yOff))
    dataset.SetProjection(proj)
    dataset.GetRasterBand(1).SetNoDataValue(0)
    dataset.GetRasterBand(1).WriteArray(vi_np)