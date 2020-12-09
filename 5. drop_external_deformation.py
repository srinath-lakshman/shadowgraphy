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

################################################################################

hard_disk = [r'F:/']

project =   [r'color_interferometry/side_view/20200114/'
             r'experiment/lower_speed_mica_run1']

folder = hard_disk[0] + '/' + project[0]

################################################################################

folder = r'E:\srinath_dhm\impact_over_thin_films\speed1\00350cs0005mum_r1\00350cs0005mum_r1'
os.chdir(folder)

# px_microns = np.loadtxt('px_microns.txt')
px_microns = 40

images = io.ImageCollection(sorted(glob.glob('*.tif'),
                            key=os.path.getmtime))
n = np.shape(images)[0]

# start_image             = int('065')
# apparent_contact_image  = int('093')
# end_image               = int('125')

start_image             = int('375')
apparent_contact_image  = int('450')
end_image               = int('525')

# plt.imshow(images[apparent_contact_image], cmap='gray')
# plt.show()

wall = 175
centerline = 130

ylimits = [wall-100, wall]
xlimits = [centerline-50, centerline+50]

plt.figure(0)
ax = plt.subplot(1,1,1)

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')

ax.set_xticks([-0.5, 49.5, 99.5])
ax.set_xticklabels(['-2', '0', '+2'])

ax.set_yticks([-0.5, 49.5, 99.5])
ax.set_yticklabels(['+4', '+2', '0'])

count = 0

for i in range(start_image, end_image+1):
    print(i)
    image_cropped = images[i][wall-100:wall, centerline-50:centerline+50]
    image_cropped_filter = filters.gaussian(image_cropped)
    edge_sobel = sobel(image_cropped_filter)
    binary = edge_sobel > 0.0075
    binary[0:20,:] = 0

    binary_x = np.argwhere(binary == True)[:,1]
    binary_y = np.argwhere(binary == True)[:,0]

    binary_filled = np.zeros(np.shape(binary))

    for y in range(np.min(binary_y), np.max(binary_y)+1):
        indices = np.argwhere(binary_y == y)
        if len(binary_x[indices]) != 0:
            x_max = np.max(binary_x[indices])
            x_min = np.min(binary_x[indices])
            binary_filled[y,x_min:x_max+1] = 1

    binary_filled_x = np.argwhere(binary_filled == True)[:,1]
    binary_filled_y = np.argwhere(binary_filled == True)[:,0]

    binary_line = np.zeros(np.shape(binary_filled))

    coordinates_left_mm = []
    coordinates_right_mm = []

    for y in range(np.min(binary_filled_y), np.max(binary_filled_y)+1):
        indices_filled = np.argwhere(binary_filled_y == y)
        if len(binary_filled_x[indices_filled]) != 0:
            x_max = np.max(binary_filled_x[indices_filled])
            x_min = np.min(binary_filled_x[indices_filled])
            binary_line[y,x_min] = 1
            binary_line[y,x_max] = 1
            coordinates_left_mm.append([(x_min-50)*px_microns/1000, (100-y)*px_microns/1000])
            if x_max != x_min:
                coordinates_right_mm.append([(x_max-50)*px_microns/1000, (100-y)*px_microns/1000])

    for x in range(np.min(binary_filled_x), np.max(binary_filled_x)+1):
        indices_filled = np.argwhere(binary_filled_x == x)
        if len(binary_filled_y[indices_filled]) != 0:
            y_max = np.max(binary_filled_y[indices_filled])
            y_min = np.min(binary_filled_y[indices_filled])
            binary_line[y_max,x] = 1
            binary_line[y_min,x] = 1

    coordinates_left_mm = np.array(coordinates_left_mm)
    coordinates_right_mm = np.array(coordinates_right_mm)
    coordinates_mm = np.vstack((np.flipud(coordinates_left_mm),coordinates_right_mm))

    ax.imshow(binary_line, cmap='gray')

    x_mm = coordinates_mm[:,0]
    y_mm = coordinates_mm[:,1]

    np.savetxt('profile_' + f'{count:03d}' + '.txt', np.column_stack([x_mm, y_mm]), fmt='%1.3f')
    plt.savefig('profile_' + f'{count:03d}' + '.png')

    count = count + 1

################################################################################
