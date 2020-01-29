import os
import matplotlib.pyplot as plt
import glob
from skimage import io
import numpy as np
from PIL import Image

################################################################################

hard_disk   = r'/media/devici/Samsung_T5/'
project     = r'srinath_dhm/impact_over_thin_films/speed2/00100cs0010mum_r4/'

f = hard_disk + project + r'100cst_10mum_reference_image'
os.chdir(f)

image_file = glob.glob('*.tif')[0]

image = io.imread(image_file)

binary = image > 1000

x_px = np.shape(image)[0]
y_px = np.shape(image)[1]

y_start = 360
y_end = 700

diff = np.zeros(y_end - y_start + 1, dtype='float')

for j in np.arange(y_start, y_end+1, 1):
    diff[int(j-y_start)] = (1 - np.mean(binary[j,:]))*x_px

# plt.figure(1)
# plt.imshow(image, cmap=plt.get_cmap('gray'))
#
# plt.figure(2)
# plt.imshow(binary, cmap=plt.get_cmap('gray'))
#
# plt.figure(3)
# plt.plot(diff)
#
# plt.show()

avg_diff = int(np.mean(diff))

length_mm = 10.0
px = (length_mm/avg_diff)*(10.0**3)

print(px)

################################################################################
