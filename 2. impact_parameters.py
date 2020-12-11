import os
import matplotlib.pyplot as plt
import glob
from skimage import io
from skimage import feature
from skimage.measure import regionprops
from skimage import filters
import skimage
import numpy as np
from PIL import Image
import math
from skimage import measure
from skimage import draw
from skimage.filters import threshold_mean
from skimage.filters import threshold_otsu
from skimage.filters import sobel
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import morphology
from skimage import color
from skimage import feature
from skimage import util
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage import segmentation

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5/'
project     = r'color_interferometry/side_view/20201208/'

################################################################################

file = hard_disk + '/' + project
os.chdir(file)
px_microns = np.loadtxt("px_microns.txt")

file = hard_disk + '/' + project + '/' + r'oil_1cSt_impact_H_4R_on_oil_350cSt_15mum_'
os.chdir(file)

fps_hz = int(tuple(open("oil_1cSt_impact_H_4R_on_oil_350cSt_15mum_.cih",'r'))[15][19:])

images = io.ImageCollection(sorted(glob.glob('*.tif'), key=os.path.getmtime))
n = len(glob.glob('*.tif')) - 1
k_start, k_end = 21, 66

# plt.subplot(1,2,1)
# plt.imshow(images[k_start],cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(images[k_end],cmap='gray')
# plt.show()

y_min, y_max = 675, 999
x_min, x_max = 130, 380
radii_min, radii_max = 95, 105

centers = np.zeros((k_end-k_start+1,2), dtype=int)
diameters = np.zeros(k_end-k_start+1, dtype=int)
time_millisec = np.arange(0,((k_end-k_start+1)*1000.0)/fps_hz,1000.0/fps_hz, dtype=float)

for k in range(k_start, k_end+1):
    j=0

    x=np.zeros(k_end-k_start+1)
    y=np.zeros(k_end-k_start+1)

    image = images[k]
    image_cropped = image[y_min:y_max+1,x_min:x_max+1]

    # image_modified = util.img_as_ubyte(image_cropped)
    image_modified = image_cropped

    #blurring image using median filter
    image_cropped_filter = filters.median(image_modified)

    #edge detection
    edge_sobel = filters.sobel(image_cropped_filter)
    # edge_sobel = feature.canny(image_cropped_filter)

    #closing
    # closing = morphology.closing(edge_sobel, morphology.disk(2))
    # morphology.disk(20)

    #thresholding
    threshold = threshold_otsu(edge_sobel)
    binary = edge_sobel > threshold

    boundary = segmentation.find_boundaries(binary, connectivity=1, mode='inner', background=0)

    #edge skeleton
    drop_skeleton = morphology.skeletonize(binary)

    label_image = measure.label(drop_skeleton==1, connectivity=2)
    image_label_overlay = color.label2rgb(label_image, image=drop_skeleton)

    print(np.shape(label_image))

    props = measure.regionprops(label_image, intensity_image=drop_skeleton)

    # [x,y] = np.array(props[0:8].centroid

    perimeter = np.zeros(label_image.max())

    for i in range(label_image.max()):
        perimeter[i] = props[i].perimeter

    index = np.argmax(perimeter)

    [x[j], y[j]] = props[index].centroid
    xy_coor = props[index].coords
    # print(props[index])
    # input('')

    # contour = measure.find_contour(labels == props[index].label, 0.5)[0]

    # M = moments(image)
    # centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

    plt.subplot(2,3,1)
    plt.imshow(image_cropped,cmap='gray')
    plt.scatter(xy_coor[:,1], xy_coor[:,0])
    plt.title('Raw image')
    plt.subplot(2,3,2)
    # plt.hist(image_cropped)
    plt.imshow(image_cropped_filter,cmap='gray')
    # plt.title('Median Filtered image')
    plt.subplot(2,3,3)
    plt.imshow(edge_sobel,cmap='gray')
    # plt.title('Sobel Edge image')
    plt.subplot(2,3,4)
    plt.imshow(binary,cmap='gray')
    # plt.title('Watershed image')
    plt.subplot(2,3,5)
    plt.imshow(drop_skeleton,cmap='gray')
    # plt.plot(peak_idx[:,1],peak_idx[:,0],'r.')
    # # plt.title('Segmentation image')
    plt.subplot(2,3,6)
    plt.imshow(label_image)
    # plt.title('Segmentation Color image')
    # plt.subplot(3,3,7)
    # plt.imshow(color.label2rgb(labels, image=image_modified, kind='avg'), cmap='gray')
    # # plt.title('Segmentation Geayscale image')
    plt.tight_layout()
    plt.show()

    j = j+1

    plt.imshow(edge_sobel, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('image_cropped' + str(k) +'.png', bbox_inches='tight', format='png')
    plt.close()

# diameter_mm = diameters[:]*(px_microns/1000)
diameter_mm = 2

plt.subplot(1,2,1)
plt.scatter(range(len(x)),x)
plt.subplot(1,2,2)
plt.scatter(range(len(y)),y)
plt.show()

xc_mm = abs(centers[:,1]-centers[0,1])*(px_microns/1000.0)
yc_mm = abs(centers[:,0]-(y_max-y_min+1))*(px_microns/1000.0)

avg_diameter_mm = np.mean(diameters[:])*(px_microns/1000)
diameter_mm_fitted = np.ones(len(time_millisec))* avg_diameter_mm

linear_params = np.polyfit(time_millisec, xc_mm, 1)
vxc_ms = linear_params[0]
xc_mm_initial = linear_params[1]
xc_mm_fitted = (time_millisec*vxc_ms) + xc_mm_initial

quadratic_params = np.polyfit(time_millisec, yc_mm, 2)
ayc_ms2 = quadratic_params[0]*2.0*(1000.0)
vyc_ms = quadratic_params[1]
yc_mm_initial = quadratic_params[2]
yc_mm_fitted = ((1/2.0)*(time_millisec**2.0)*ayc_ms2*(1/1000.0)) + (time_millisec*vyc_ms) + yc_mm_initial
vy_impact_ms = (time_millisec[-1]*ayc_ms2*(1/1000.0)) + vyc_ms
fall_height_mm = abs(yc_mm_fitted[0] - yc_mm_fitted[-1])
fall_time_ms = abs(time_millisec[0] - time_millisec[-1])

plt.figure(figsize=(15, 10))

plt.subplot(2,1,1)
plt.scatter(time_millisec,diameter_mm, marker='.', color='black')
plt.plot(time_millisec,diameter_mm_fitted, linestyle='--', color='black')
plt.xlabel('$t$ $[ms]$')
plt.ylabel('$D_{w}(t)$ $[mm]$')
plt.ylim(0,2.0*avg_diameter_mm)
plt.title(r'$D_{w}$ $\approx$ ' + str(round(avg_diameter_mm,3)) + r' $mm$, $v_{w}$ $\approx$ ' + str(abs(round(vy_impact_ms,3))) + r' $m/s$, $\Delta H_{w}$ $\approx$ ' + str(abs(round(fall_height_mm,3))) + r' $mm$, $\Delta t_{w}$ $\approx$ ' + str(abs(round(fall_time_ms,3))) + ' $ms$')

plt.subplot(2,2,3)
plt.scatter(time_millisec,xc_mm, marker='.', color='black')
plt.plot(time_millisec,xc_mm_fitted, linestyle='--', color='black')
plt.xlabel('$t$ $[ms]$')
plt.ylabel('$x_{c}(t)$ $[mm]$')
plt.title(r'$|x_{c}(0)|$ $=$ ' + str(abs(round(xc_mm_initial,3))) + ' $mm$, $|vx_{c}|$ $=$ ' + str(abs(round(vxc_ms,3))) +' $m/s$')

plt.subplot(2,2,4)
plt.scatter(time_millisec,yc_mm, marker='.', color='black')
plt.plot(time_millisec,yc_mm_fitted, linestyle='--', color='black')
plt.xlabel('$t$ $[ms]$')
plt.ylabel('$y_{c}(t)$ $[mm]$')
plt.title(r'$|y_{c}(0)|$ $=$ ' + str(abs(round(yc_mm_initial,3))) + ' $mm$, $|vy_{c}(0)|$ $=$ ' + str(abs(round(vyc_ms,3))) + ' $m/s$, $|ay_{c}|$ $=$ ' + str(abs(round(ayc_ms2,3))) + ' $m/s^2$')

plt.savefig('impact_parameters.pdf', format='pdf')

txt_file = open("impactspeed_sideview_info.txt","w")
txt_file.write(f"1 pixel = {px_microns} microns\n")
txt_file.write(f"Recording speed = {fps_hz} Hz\n")
txt_file.write(f"Image number = {k_start, k_end}\n")
txt_file.write(f"Y-limits = {y_min, y_max} pixels\n")
txt_file.write(f"X-limits = {x_min, x_max} pixels\n")
txt_file.write(f"Radii limits = {radii_min, radii_max} pixels\n")
txt_file.close()

################################################################################
