import os
import matplotlib.pyplot as plt
import glob
from skimage import io
import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage import filters
from FUNC_ import rectangular_fit

# import sys
# sys.path.append(r'/home/devici/github/functions')
#
# import FUNC_

################################################################################

foldername              = r'D:\harddisk_file_contents\color_interferometry\side_view\20201212'
image_filename          = r'reference_lengthscale.tif'
approximate_center      = [0,0]
approximate_crop        = 0
approximate_threshold   = 0
approximate_radii       = [0,0]

px_microns = rectangular_fit(\
                                foldername      = foldername, \
                                image_filename  = image_filename, \
                                center          = approximate_center, \
                                crop            = approximate_crop, \
                                threshold       = approximate_threshold, \
                                radii           = approximate_radii)

image = io.imread(image_filename)

directory = r'/media/devici/328C773C8C76F9A5/color_interferometry/side_view/20201208/old'
image_filename = r'reference_lengthscale'

os.chdir(directory)

image = io.imread(image_filename + '.tif')
bit_depth = 12

image_conv = image * (((2**16)-1)/((2**bit_depth)-1))

gray_filter = filters.gaussian(image_conv)
edge_sobel = sobel(gray_filter)
threshold = int(threshold_otsu(edge_sobel))
binary = edge_sobel > threshold

fig_rows = 2
fig_columns = 2

fig1 = plt.figure(1, figsize=(4,4))
ax1 = plt.subplot(fig_rows,fig_columns,1)
ax1.imshow(image_conv, cmap='gray')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title("Grayscale Image, 16 bit")

ax2 = plt.subplot(fig_rows,fig_columns,2)
ax2.imshow(gray_filter, cmap='gray')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Gaussian fliter")

ax3 = plt.subplot(fig_rows,fig_columns,3)
ax3.imshow(edge_sobel, cmap='gray')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_title("Edge Sobel")

ax4 = plt.subplot(fig_rows,fig_columns,4)
ax4.imshow(binary, cmap='gray')
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_title(f"Binary Image, Threshold = {threshold}")

plt.show(block=False)
plt.show()

x_px, y_px = list(np.shape(image))

y_extents = np.array(input('Vertical y-extents = ').split(',')).astype('int')

ax1.axhline(y=y_extents[0], linestyle='--', color='red')
ax1.axhline(y=y_extents[1], linestyle='--', color='red')

ax2.axhline(y=y_extents[0], linestyle='--', color='red')
ax2.axhline(y=y_extents[1], linestyle='--', color='red')

ax3.axhline(y=y_extents[0], linestyle='--', color='red')
ax3.axhline(y=y_extents[1], linestyle='--', color='red')

ax4.axhline(y=y_extents[0], linestyle='--', color='red')
ax4.axhline(y=y_extents[1], linestyle='--', color='red')

y_start = min(y_extents[0], y_extents[1])
y_end = max(y_extents[0], y_extents[1])

y_line = np.arange(y_start, y_end+1, 1)
n = len(y_line)
# diff = np.zeros(n)
coor_left = np.zeros((n,2))
coor_right = np.zeros((n,2))

for i in range(n):
    coor_left[i] = np.mean(np.where(binary[y_line[i],0:int(x_px/2)]==1)), y_line[i]
    coor_right[i] = int(x_px/2) + np.mean(np.where(binary[y_line[i],int(x_px/2):x_px]==1)), y_line[i]

x_left_avg = np.mean(coor_left[:,0])
x_right_avg = np.mean(coor_right[:,0])

ax1.plot([x_left_avg, x_left_avg], [y_start, y_end], linestyle='--', color='red')
ax1.plot([x_right_avg, x_right_avg], [y_start, y_end], linestyle='--', color='red')

ax2.plot([x_left_avg, x_left_avg], [y_start, y_end], linestyle='--', color='red')
ax2.plot([x_right_avg, x_right_avg], [y_start, y_end], linestyle='--', color='red')

ax3.plot([x_left_avg, x_left_avg], [y_start, y_end], linestyle='--', color='red')
ax3.plot([x_right_avg, x_right_avg], [y_start, y_end], linestyle='--', color='red')

ax4.plot([x_left_avg, x_left_avg], [y_start, y_end], linestyle='--', color='red')
ax4.plot([x_right_avg, x_right_avg], [y_start, y_end], linestyle='--', color='red')

plt.show(block=False)

length_px = int(abs(x_right_avg-x_left_avg))
length_mm = np.loadtxt("reference_lengthscale_mm.txt")
px_microns = round((length_mm/length_px)*(10.0**3),3)

input(f"1 pixel = {px_microns} microns")

np.savetxt("px_microns.txt", [px_microns], fmt='%0.3f')

################################################################################
