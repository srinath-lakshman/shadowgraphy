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

lengthscale_foldername  = r'D:/harddisk_file_contents/color_interferometry/side_view/20201212'
lengthscale_file        = r'px_microns.txt'

os.chdir(lengthscale_foldername)
px_microns = np.loadtxt("px_microns.txt")

################################################################################

foldername = r'D:\harddisk_file_contents\color_interferometry\side_view\20201212\oil_1cSt_impact_H_4R_on_100cSt_10mum_run1_'

os.chdir(foldername)

# hard_disk   = r'D:/harddisk_file_contents/'
# project     = r'color_interferometry/side_view/20201212/oil_1cSt_impact_H_4R_on_100cSt_10mum_run1_'

################################################################################

# file = hard_disk + '/' + project
# os.chdir(file)
# px_microns = np.loadtxt("px_microns.txt")
# px_microns = 10.291

# file = hard_disk + '/' + project + '/' + r'oil_1cSt_impact_H_4R_on_oil_350cSt_15mum_'
# file = hard_disk + '/' + project + '/'
# os.chdir(file)

fps_hz = int(tuple(open("oil_1cSt_impact_H_4R_on_100cSt_10mum_run1_.cih",'r'))[15][19:])

images = io.ImageCollection(sorted(glob.glob('*.tif'), key=os.path.getmtime))
n = len(glob.glob('*.tif')) - 1

k_start, k_end = 18, 117

# plt.subplot(1,2,1)
# plt.imshow(images[k_start],cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(images[k_end],cmap='gray')
# plt.show()

y_wall, y_top = 1000, 680

x_center = int(np.shape(images)[2]/2)
y_min, y_max = y_wall-360, y_wall+100
x_min, x_max = x_center-150, x_center+150

centers = np.zeros((k_end-k_start+1,2), dtype=int)
diameters = np.zeros(k_end-k_start+1, dtype=int)
volume = np.zeros((k_end-k_start+1), dtype=int)
time_millisec = np.arange(0,((k_end-k_start+1)*1000.0)/fps_hz,1000.0/fps_hz, dtype=float)
# time_millisec = time_millisec - time_millisec[-1]

threshold = 500
radius = 35

for k in range(k_start, k_end+1):

    print(k)
    image = images[k]
    image_cropped = image[y_min:y_max+1,x_min:x_max+1]
    image_cropped_filter = filters.median(image_cropped)
    binary = image_cropped_filter < threshold
    binary[y_wall-y_min:,:] = 0
    binary[:y_top-y_min,:] = 0
    closing = morphology.closing(binary, morphology.disk(radius))
    boundary = segmentation.find_boundaries(closing, connectivity=1, mode='outer', background=0)
    indices = np.transpose(np.where(boundary == 1))
    indices[:,[0,1]] = indices[:,[1,0]]
    indices = indices[indices[:, 1].argsort()]
    n = np.shape(indices)[0]

    axis = np.mean(indices[:,0])
    vol = (np.pi/2)*np.trapz(np.power(indices[:,0]-axis,2))
    xx = axis
    yy = (np.pi/2)*np.trapz(indices[:,1]*np.power(indices[:,0]-axis,2))/vol

    centers[k-k_start] = [axis, yy]
    volume[k-k_start] = vol

    plt.subplot(2,3,1)
    plt.imshow(image_cropped,cmap='gray')
    plt.title('Raw image')
    plt.axis('off')
    plt.subplot(2,3,2)
    plt.imshow(image_cropped_filter,cmap='gray')
    plt.title('Median Filtered image')
    plt.axis('off')
    plt.subplot(2,3,3)
    plt.imshow(binary,cmap='gray')
    plt.title('Binary otsu image')
    plt.axis('off')
    plt.subplot(2,3,4)
    plt.imshow(closing,cmap='gray')
    plt.title('Closing image')
    plt.axis('off')
    plt.subplot(2,3,5)
    plt.imshow(boundary,cmap='gray')
    plt.title('Boundary image')
    plt.axis('off')
    plt.subplot(2,3,6)
    plt.imshow(image_cropped,cmap='gray')
    plt.scatter(indices[:,0], indices[:,1])
    plt.scatter(xx, yy)
    plt.title('Center image')
    plt.axis('off')
    plt.savefig('image_cropped' + str(k) +'.png', bbox_inches='tight', format='png')
    plt.close()
    # plt.show()

xc_mm = abs(centers[:,0]-centers[0,0])*(px_microns/1000.0)
yc_mm = abs(centers[:,1]-(y_wall-y_min+1))*(px_microns/1000.0)

diameters = np.power((6*volume)/np.pi,1/3)
diameter_mm = diameters*(px_microns/1000)
avg_diameter_mm = np.mean(diameter_mm)
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
plt.scatter(time_millisec,diameter_mm/2, marker='.', color='black')
plt.plot(time_millisec,diameter_mm_fitted/2, linestyle='--', color='black')
plt.xlabel('$t$ $[ms]$')
plt.ylabel('$R_{w}(t)$ $[mm]$')
plt.ylim(0.5,1.5)
plt.title(r'$R_{w}$ $\approx$ ' + str(round(avg_diameter_mm/2,3)) + r' $mm$, $v_{w}$ $\approx$ ' + str(abs(round(vy_impact_ms,3))) + r' $m/s$, $\Delta H_{w}$ $\approx$ ' + str(abs(round(fall_height_mm,3))) + r' $mm$, $\Delta t_{w}$ $\approx$ ' + str(abs(round(fall_time_ms,3))) + ' $ms$')

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
# txt_file.write(f"Radii limits = {radii_min, radii_max} pixels\n")
txt_file.close()

################################################################################
