import os
import matplotlib.pyplot as plt
import glob
from skimage import io
import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage import filters

################################################################################

hard_disk   = r'/media/devici/Samsung_T5'
project     = r'srinath_dhm/impact_over_thin_films/speed1/00100cs0010mum_r4'

################################################################################

f = hard_disk + '/' + project + '/' + r'00100cs0010mum_r4'
os.chdir(f)

image_name = r"screw_diameter_mm"

image = io.imread(image_name + ".tif")
bit_depth = int(tuple(open(image_name + ".cih",'r'))[26][12:])

image_conv = image * (((2**16)-1)/((2**bit_depth)-1))

gray_filter = filters.gaussian(image_conv)
edge_sobel = sobel(gray_filter)
threshold = int(threshold_otsu(edge_sobel))
binary = edge_sobel > threshold

fig_rows = 2
fig_columns = 2

fig1 = plt.figure(1, figsize=(8,8))
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

length_px = abs(x_right_avg-x_left_avg)
length_mm = np.loadtxt("screw_diameter_mm.txt")
px_microns = round((length_mm/length_px)*(10.0**3),3)

input(f"1 pixel = {px_microns} microns")
np.savetxt("px_microns.txt", [px_microns], fmt='%0.3f')

################################################################################
