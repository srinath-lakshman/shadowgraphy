import numpy as np
import os
import glob
import cv2
import sys
import re
import math

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image

from scipy import stats
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit

import skimage
from skimage import io
from skimage import util
from skimage import color
from skimage import feature
from skimage import segmentation

from skimage import filters
from skimage.filters import sobel
from skimage.filters import threshold_mean
from skimage.filters import threshold_otsu
from skimage.filters import threshold_minimum

from skimage.transform import hough_circle
from skimage.transform import hough_circle_peaks

from skimage import draw
from skimage.draw import circle_perimeter

from skimage import graph
from skimage.graph import route_through_array

from skimage import morphology
from skimage.morphology import skeletonize

from skimage import measure
from skimage.measure import regionprops

################################################################################

def vertical_trajectory(t=[None], r=[None]):

    a = -9.81
    r_mod = r - ((a/(2.0*1000.0))*(t**2))
    linear_params = np.polyfit(t, r_mod, 1)
    rs = ((a/(2.0*1000.0))*(t[0]**2)) + \
         (linear_params[0]*(t[0]**1)) + \
         (linear_params[1]*(t[0]**0))
    vs = ((a/1000.0)*t[0]) + linear_params[0]

    if max(t)<=0:
        tt = t[-1]
    else:
        tt = -(linear_params[0]*1000.0)/a
        print(tt)

    re = ((a/(2.0*1000.0))*(tt**2)) + \
         (linear_params[0]*(tt**1)) + \
         (linear_params[1]*(tt**0))
    ve = ((a/1000.0)*tt) + linear_params[0]
    rfit = ((a/(2.0*1000.0))*(t**2)) + \
           (linear_params[0]*(t**1)) + \
           (linear_params[1]*(t**0))
    d = tt - t[0]

    return rs, vs, re, ve, rfit, d

################################################################################

os.chdir(r'D:\harddisk_file_contents\color_interferometry\side_view\20201212\oil_1cSt_impact_H_4R_on_100cSt_10mum_run1_\info')

x = np.loadtxt('xc_mm.txt')
y = np.loadtxt('yc_mm.txt')
t = np.loadtxt('time_ms.txt')

# m = 5
# n = 105

xc_mm = x
yc_mm = y
t_ms = t

plt.figure(1)
plt.scatter(t_ms, yc_mm, marker='.', color='black')

yr, vr, ye, ve, yfit, d = vertical_trajectory(t_ms, yc_mm)
print('yr, vr = ' + f'{yr:.3f} mm, {vr:.3f} m/s')
print('ye, ve = ' + f'{ye:.3f} mm, {ve:.3f} m/s')
print('t = ' + f'{d:.3f} ms')
plt.plot(t_ms,yfit, linestyle='--', color='red')

plt.show()

################################################################################
