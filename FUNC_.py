from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from scipy import stats
from skimage import color
from skimage import io
import re
import math
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage.filters import threshold_minimum
from skimage import filters
from skimage.transform import hough_circle
from skimage.draw import circle_perimeter
from skimage.graph import route_through_array
from skimage import morphology
from skimage.morphology import skeletonize

########################################

def rectangular_fit(foldername='', image_filename='', vertical_crop=[0,0]):

    os.chdir(foldername)
    image = io.imread(image_filename)

    image_cropped = image[vertical_crop[0]:vertical_crop[1],:]
    threshold = threshold_otsu(image_cropped)
    binary = image_cropped < threshold
    z = np.count_nonzero(binary)

    length_px = int(z/(vertical_crop[1]-vertical_crop[0]))
    length_mm = np.loadtxt("reference_lengthscale_mm.txt")
    px_microns = round((length_mm/length_px)*(10.0**3),3)

    print(f"1 pixel = {px_microns} microns")

    fig = plt.figure(1,figsize=(5,4))

    ax1 = plt.subplot(1,2,1)
    ax1.imshow(image, cmap='gray')

    ax2 = plt.subplot(2,2,2)
    ax2.imshow(image_cropped, cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = plt.subplot(2,2,4)
    ax3.imshow(~binary, cmap='gray')
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    plt.show()

    np.savetxt("px_microns.txt", [px_microns], fmt='%0.3f')

    return px_microns

########################################
