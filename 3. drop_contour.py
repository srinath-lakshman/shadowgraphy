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

################################################################################

hard_disk   = r'F:/'
project     = r'color_interferometry\side_view\20200114\experiment\lower_speed_mica_run1'

################################################################################

folder = hard_disk + '/' + project
os.chdir(folder)

px_microns = 24.75

images = io.ImageCollection(sorted(glob.glob('*.tif'), key=os.path.getmtime))
n = len(glob.glob('*.tif')) - 1

start_image = int('078')
end_image   = int('108')

# start_image = int('101')

# image_cropped = image[y_min:y_max+1,x_min:x_max+1]
# image_cropped_filter = filters.gaussian(image_cropped)
# edge_sobel = sobel(image_cropped_filter)
# threshold = threshold_otsu(edge_sobel)
# binary = edge_sobel > threshold

y_wall = 543
x_center = 367

ylimits = [y_wall-100, y_wall]
xlimits = [x_center-100, x_center+100]

fig, ax = plt.subplots(1,1)

for i in range(start_image, end_image+1):
    image_cropped = images[i][ylimits[0]:ylimits[1], xlimits[0]:xlimits[1]]
    image_cropped_filter = filters.gaussian(image_cropped)
    edge_sobel = sobel(image_cropped_filter)
    threshold = threshold_otsu(edge_sobel)
    binary = edge_sobel > threshold

    binary[80:100,75:125] = 0

    binary_x = np.argwhere(binary == True)[:,1]
    binary_y = np.argwhere(binary == True)[:,0]

    coor_left = []
    for y_avg in np.arange(np.min(binary_y), np.max(binary_y)+1,1):
        indices = np.argwhere(binary_y == y_avg)
        x_avg = binary_x[indices]
        x_left_values = x_avg[np.argwhere(binary_x[indices] <= 100)[:,0]]
        if len(x_left_values) != 0:
            x_left_min = np.min(x_left_values)
            coor_left.append([x_left_min, y_avg])

    coor_right = []
    for y_avg in np.arange(np.max(binary_y), np.min(binary_y)-1,-1):
        indices = np.argwhere(binary_y == y_avg)
        x_avg = binary_x[indices]
        x_right_values = x_avg[np.argwhere(binary_x[indices] >= 100)[:,0]]
        if len(x_right_values) != 0:
            x_right_max = np.max(x_right_values)
            coor_right.append([x_right_max, y_avg])
    coor_left = np.array(coor_left)
    coor_right = np.array(coor_right)
    coor = np.vstack((np.flipud(coor_left),np.flipud(coor_right)))
    p10 = np.poly1d(np.polyfit(coor[:,1], coor[:,0], 10))
    p10_left = np.poly1d(np.polyfit(coor_left[:,1], coor_left[:,0], 10))
    p10_right = np.poly1d(np.polyfit(coor_right[:,1], coor_right[:,0], 10))

    # ax.imshow(image_cropped,cmap='gray')
    # ax.plot(p10_right(coor_right[:,1]),coor_right[:,1],marker='.',color='blue')

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # sampling_y = np.linspace(np.min(coor_right[:,1]), np.max(coor_right[:,1]), 100)
    # sampling_x = p10_right(sampling_y)
    # sampling_x = np.flipud(sampling_x)
    # sampling_y = np.flipud(sampling_y)
    # x_microns = (sampling_x - 107)*px_microns
    # y_microns = (99 - sampling_y)*px_microns
    x_microns = (p10_right(coor_right[:,1]) - 107)*px_microns
    y_microns = (99 - coor_right[:,1])*px_microns
    x_mm = x_microns/1000
    y_mm = y_microns/1000
    ax.scatter(x_mm, y_mm, marker='.',color='black')
    # ax.axis('equal')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.25,2)
    ax.set_ylim(-0.25,2.25)
    ax.set_xticks([0, 0.50, 1, 1.50, 2])
    ax.set_yticks([0, 0.50, 1, 1.50, 2])
    ax.set_xlabel(r'r [$mm$]')
    ax.set_ylabel(r'h [$mm$]')
    ax.axhline(y=0,linestyle='--',color='black')
    ax.axvline(x=0,linestyle='--',color='black')
    # plt.tight_layout()
    plt.savefig('drop_outer_profile_' + f'{i:06d}' + '.png')
    np.savetxt('outer_profile_' + f'{i:06d}' + '.txt', np.column_stack([x_mm, y_mm]), fmt='%1.3f')
    plt.cla()

################################################################################
