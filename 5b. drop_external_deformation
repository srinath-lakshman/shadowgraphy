import os
import matplotlib.pyplot as plt
import glob
from skimage import io
import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage import filters
from skimage import measure
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

################################################################################

directory1 = r'E:/color_interferometry/side_view/20200520/experiment/'
directory2 = r'needle_tip_45_mm_from_surface/impact_on_glass_r1/'

os.chdir(directory1)
px_microns = np.loadtxt(r'px_microns.txt')

os.chdir(directory2)
images = io.ImageCollection(sorted(glob.glob('*.tif'),
                            key=os.path.getmtime))
n = np.shape(images)[0]

start_image             = int('028') - 1
apparent_contact_image  = int('114') - 1
end_image               = (2*apparent_contact_image) - start_image

# plt.imshow(images[apparent_contact_image], cmap='gray')
# plt.show()

wall = 800
symmetry = 500
halfwidth = 150
height = 400

# plt.subplot(1,3,1)
# plt.imshow(images[start_image], cmap='gray')
#
# plt.subplot(1,3,2)
# plt.imshow(images[apparent_contact_image], cmap='gray')
#
# plt.subplot(1,3,3)
# plt.imshow(images[end_image], cmap='gray')
# plt.show()

ylimits = [wall-height, wall]
xlimits = [symmetry-halfwidth, symmetry+halfwidth]

# plt.figure(1)
# ax = plt.subplot(1,1,1)
#
# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
#
# ax.set_xlim([0-0.5, (halfwidth*2) -0.5])
# ax.set_ylim([height-0.5, 0-0.5])
#
# ax.set_xticks([])
# ax.set_yticks([])

for i in range(start_image, end_image+1):
    print(i)
    image_cropped = images[i][wall-height:wall, symmetry-halfwidth:symmetry+halfwidth]
    image_cropped_filter = filters.gaussian(image_cropped)
    edge_sobel = sobel(image_cropped_filter)

    plt.imshow(edge_sobel, cmap='gray')
    # plt.axis('off')
    plt.tight_layout()
    plt.savefig('drop_profile_' + f'{i:03d}' + '.tif', dpi = 96)
    # plt.savefig('drop_profile_' + f'{i:03d}' + '.png')
    plt.show()

################################################################################
