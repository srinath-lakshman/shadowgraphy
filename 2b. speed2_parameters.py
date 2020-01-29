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
from skimage import color

################################################################################

hard_disk   = r'/media/devici/Samsung_T5/'
project     = r'srinath_dhm/impact_over_thin_films/speed2/00100cs0010mum_r4/'

f = hard_disk + project + r'0100cs0010mum_r4'
os.chdir(f)

px_microns = 16.155
fps_hz = 5000.0

k_start = 19
k_end = 186

y_min = 129
y_max = 719

x_min = 250
x_max = 500

images = io.ImageCollection(sorted(glob.glob('*.tif'), key=os.path.getmtime))
n = len(glob.glob('*.tif'))

centers = np.zeros((k_end-k_start+1,2), dtype=int)
diameters = np.zeros(k_end-k_start+1, dtype=int)
time_millisec = np.arange(0,((k_end-k_start+1)*1000.0)/fps_hz,1000.0/fps_hz,dtype=float)

for k in range(k_start, k_end+1):
    image = images[k]
    image_cropped = image[y_min:y_max+1,x_min:x_max+1]
    image_cropped_filter = filters.gaussian(image_cropped)
    edge_sobel = sobel(image_cropped_filter)
    threshold = threshold_otsu(edge_sobel)
    binary = edge_sobel > threshold

    hough_radii = np.arange(60, 71)
    hough_res = hough_circle(binary, hough_radii)
    ridx, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)

    image_cropped = color.gray2rgb(image_cropped*float(((2**8)-1.0)/((2**12)-1.0))).astype(int)
    rr, cc = circle_perimeter(r,c,hough_radii[ridx])
    image_cropped[rr, cc] = (255, 0, 0)
    image_cropped[r,c] = (0, 0, 255)

    centers[int(k-k_start)] = (r,c)
    diameters[int(k-k_start)] = 2.0*hough_radii[ridx]

    plt.imshow(image_cropped, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('image_cropped' + str(k) +'.png', bbox_inches='tight', format='png')
    plt.close()

diameter_mm = diameters[:]*(px_microns/1000)
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

plt.savefig('info', format='pdf')

################################################################################
