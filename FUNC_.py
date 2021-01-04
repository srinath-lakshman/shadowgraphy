from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from scipy import stats
from skimage import color
from skimage import io
import re
import math
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage import filters
from skimage.transform import hough_circle
from skimage.draw import circle_perimeter
from skimage.graph import route_through_array

########################################

def rectangular_fit(foldername='', image_filename='', center=[0,0], crop=0, threshold=0, radii=[0,0]):

    os.chdir(foldername)
    image = io.imread(image_filename)
    y_px, x_px = list(np.shape(image))
    # bit_depth = 12

    gray_filter = filters.gaussian(image)
    edge_sobel = sobel(gray_filter)
    threshold = int(threshold_otsu(edge_sobel))
    binary = edge_sobel > threshold

    fig_rows = 2
    fig_columns = 2

    fig1 = plt.figure(1, figsize=(2,5))
    ax1 = plt.subplot(fig_rows,fig_columns,1)
    ax1.imshow(image, cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax1.set_title("Grayscale Image, 12 bit")

    ax2 = plt.subplot(fig_rows,fig_columns,2)
    ax2.imshow(gray_filter, cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax2.set_title("Gaussian fliter")

    ax3 = plt.subplot(fig_rows,fig_columns,3)
    ax3.imshow(edge_sobel, cmap='gray')
    ax3.set_xticks([])
    ax3.set_yticks([])
    # ax3.set_title("Edge Sobel")

    ax4 = plt.subplot(fig_rows,fig_columns,4)
    ax4.imshow(binary, cmap='gray')
    ax4.set_xticks([])
    ax4.set_yticks([])
    # ax4.set_title(f"Binary Image, Threshold = {threshold}")

    plt.tight_layout()
    plt.show()
    input()

    print("########## CENTERING ##########")
    print("Gray Image")

    gray = color.rgb2gray(image)*float((2**16)-1)
    x_res, y_res = list(np.shape(gray))
    xc, yc = (x_res-1)/2, (y_res-1)/2

    plt.close()
    f = plt.figure(1, figsize=(6,4))
    ax1 = plt.subplot(2,3,1)
    ax1.imshow(gray, cmap='gray')
    ax1.axvline(x = xc, linestyle='-', color='black')
    ax1.axhline(y = yc, linestyle='-', color='black')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Image')

    ax2 = plt.subplot(2,3,2)
    ax2.set_title('Centered Image')
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = plt.subplot(2,3,3)
    ax3.set_title('Cropped Image')
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = plt.subplot(2,3,4)
    ax4.set_title('Edge Image')
    ax4.set_aspect('equal')
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = plt.subplot(2,3,5)
    ax5.set_title('Binary Image')
    ax5.set_aspect('equal')
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax6 = plt.subplot(2,3,6)
    ax6.set_title('Circlefit Image')
    ax6.set_aspect('equal')
    ax6.set_xticks([])
    ax6.set_yticks([])

    plt.show(block=False)

    if np.array_equal( center, np.zeros((1,2)) ):
        center = np.array(input('Approximate center = ').split(',')).astype('int')

    radius = int(min(center[0], center[1], x_res-1-center[0], y_res-1-center[1]))
    gray_centered = gray[center[1]-radius:center[1]+radius+1, center[0]-radius:center[0]+radius+1]

    ax2.imshow(gray_centered, cmap='gray', extent=[-radius,+radius,-radius,+radius])
    ax2.set_xlim(-radius,+radius)
    ax2.set_ylim(-radius,+radius)
    ax2.axvline(x = 0, linestyle='-', color='black')
    ax2.axhline(y = 0, linestyle='-', color='black')
    plt.show(block=False)

    print('Centering done!!')
    print('#################################\n')

    print("############# CROP #############")
    print("Centered Image")

    if crop == 0:
        crop = int(input("Crop distance from center = "))

    gray_crop = gray_centered[radius-crop:radius+crop+1,radius-crop:radius+crop+1]

    ax3.imshow(gray_crop, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    plt.show(block=False)

    print('Crop done!!')
    print('#################################\n')

    gray_filter = filters.gaussian(gray_crop)
    edge_sobel = sobel(gray_filter).astype('int')
    min_val, max_val = edge_sobel.min(), edge_sobel.max()

    print("########### THRESHOLD ###########")
    print("Edge Sobel Image")
    print(f"[min,max] = [{min_val},{max_val}]")

    ax4.imshow(edge_sobel, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])

    binary = edge_sobel < int(threshold_otsu(edge_sobel))

    ax5.imshow(binary, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    plt.show(block=False)

    if threshold == 0:
        threshold = int(input("Threshold = "))

    if threshold > 0:
        binary = edge_sobel > abs(threshold)
    else:
        binary = edge_sobel < abs(threshold)

    ax5.cla()
    ax5.imshow(binary, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    ax5.set_title('Binary Image')
    ax5.set_aspect('equal')
    ax5.set_xlim(-crop, +crop)
    ax5.set_ylim(-crop, +crop)
    ax5.set_xticks([])
    ax5.set_yticks([])
    plt.show(block=False)

    print('Threshold done!!')
    print('#################################\n')

    print("############ RADIUS #############")
    print("Binary Image")

    if np.array_equal( radius, np.zeros((1,2)) ):
        radii = np.array(input('Radius extents = ').split(',')).astype('int')

    hough_radii = np.arange(radii[0], radii[1], 1)
    hough_res = hough_circle(binary, hough_radii)
    ridx, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    x_circle_center = c
    y_circle_center = r
    rr, cc = circle_perimeter(r,c,hough_radii[ridx])
    x_circle_perimeter = cc
    y_circle_perimeter = rr

    print('Circle fit done!!')
    print('#################################\n')

    ax5.scatter(x_circle_center-crop, -(y_circle_center-crop), marker='x', color='red')
    ax5.scatter(x_circle_perimeter-crop, -(y_circle_perimeter-crop), marker='.', color='red')

    # plt.figure(2)
    # plt.imshow(binary, cmap='gray')
    # plt.scatter(x_circle_center, y_circle_center, marker='x', color='red')
    # plt.scatter(x_circle_perimeter, y_circle_perimeter, marker='.', color='red')
    # plt.show()

    delta_x = center[0] - crop
    delta_y = center[1] - crop

    ax6.imshow(gray, cmap='gray')
    ax6.scatter(x_circle_center+delta_x, y_circle_center+delta_y, marker='x', color='black')
    ax6.scatter(x_circle_perimeter+delta_x, y_circle_perimeter+delta_y, marker='.', color='black')

    # plt.figure(2)
    # plt.subplot(1,2,1)
    # plt.imshow(binary, cmap='gray')
    # plt.scatter(x_circle_center, y_circle_center, marker='x', color='red')
    # plt.scatter(y_circle_perimeter, x_circle_perimeter, marker='.', color='red')
    #
    # plt.subplot(1,2,2)
    # plt.imshow(binary, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    # plt.scatter(x_circle_center-crop, -(y_circle_center-crop), marker='x', color='red')
    # plt.scatter(y_circle_perimeter-crop, -(x_circle_perimeter-crop), marker='.', color='red')

    center_px = [x_circle_center+delta_x, y_circle_center+delta_y]
    radius_px = hough_radii[ridx]
    radius_max_px = int(round(min(center_px[0], center_px[1], x_res-1-center_px[0], y_res-1-center_px[1])))

    print(f"center = {center_px}")
    print(f"radius = {radius_px}")
    print(f"radius max = {radius_max_px}")

    plt.show()

    return center_px, radius_px, radius_max_px
