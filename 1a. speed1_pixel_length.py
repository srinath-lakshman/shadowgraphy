import os
import matplotlib.pyplot as plt
import glob
from skimage import io
import numpy as np
from PIL import Image

################################################################################

hard_disk   = r'/media/devici/Samsung_T5/'
project     = r'srinath_dhm/impact_over_thin_films/speed1/00100cs0010mum_r4/'

f = hard_disk + project + r'00100cs0010mum_r4'
os.chdir(f)

image = io.imread(r"screw_diameter_mm.tif")

binary = image > 1000

x_px = np.shape(image)[0]
y_px = np.shape(image)[1]

y_start = 360
y_end = 700

diff = np.zeros(y_end - y_start + 1, dtype='float')

for j in np.arange(y_start, y_end+1, 1):
    diff[int(j-y_start)] = (1 - np.mean(binary[j,:]))*x_px

f = plt.figure(1)
ax1 = plt.subplot(1,3,1)
ax1.imshow(image, cmap=plt.get_cmap('gray'))

ax2 = plt.subplot(1,3,2)
ax2.imshow(binary, cmap=plt.get_cmap('gray'))

ax3 = plt.subplot(1,3,3)
ax3.plot(diff)
ax3.axis('equal')

# f.set_aspect('equal')
plt.show(block=False)

avg_diff = int(np.mean(diff))

length_mm = np.loadtxt("screw_diameter_mm.txt")
px = round((length_mm/avg_diff)*(10.0**3),3)

input(px)


################################################################################
