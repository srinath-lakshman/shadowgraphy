import numpy as np
import os
import glob
import cv2
import sys
import re
import math

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image

from scipy import stats
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_fill_holes
from scipy.optimize import curve_fit

import skimage
from skimage import io
from skimage import util
from skimage import color
from skimage import feature
from skimage import segmentation

from skimage import filters
from skimage.filters import sobel
from skimage.filters import threshold_mean
from skimage.filters import threshold_otsu
from skimage.filters import threshold_minimum

from skimage.transform import hough_circle
from skimage.transform import hough_circle_peaks

from skimage import draw
from skimage.draw import circle_perimeter

from skimage import graph
from skimage.graph import route_through_array

from skimage import morphology
from skimage.morphology import skeletonize

from skimage import measure
from skimage.measure import regionprops

################################################################################

def rectangular_fit(foldername='', image_filename='', vertical_crop=[0,0]):

    os.chdir(foldername)
    image = io.imread(image_filename)

    image_cropped = image[vertical_crop[0]:vertical_crop[1],:]
    threshold = threshold_otsu(image_cropped)
    binary = image_cropped < threshold
    z = np.count_nonzero(binary)

    length_px = int(z/(vertical_crop[1]-vertical_crop[0]))
    length_mm = np.loadtxt("reference_lengthscale_mm.txt")
    px_microns = round((length_mm/length_px)*(10.0**3),3)

    print(f"1 pixel = {px_microns} microns")

    # fig = plt.figure(1,figsize=(5,4))
    #
    # ax1 = plt.subplot(1,2,1)
    # ax1.imshow(image, cmap='gray')
    #
    # ax2 = plt.subplot(2,2,2)
    # ax2.imshow(image_cropped, cmap='gray')
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    #
    # ax3 = plt.subplot(2,2,4)
    # ax3.imshow(~binary, cmap='gray')
    # ax3.set_xticks([])
    # ax3.set_yticks([])
    #
    # plt.tight_layout()
    # plt.show()

    np.savetxt("px_microns.txt", [px_microns], fmt='%0.3f')

    return px_microns

################################################################################

def lengthscale_info(lengthscale_foldername='', lengthscale_file=''):

    os.chdir(lengthscale_foldername)
    px_microns = np.loadtxt(lengthscale_file)

    return px_microns

################################################################################

def impact_info(    folder          = '', \
                    drop            = '', \
                    film            = '', \
                    hf_ideal_mum    = None, \
                    start_frame     = None, \
                    end_frame       = None, \
                    ceiling         = None, \
                    wall            = None, \
                    free_surface    = None, \
                    threshold       = [None, None], \
                    px_microns      = None):

    os.chdir(folder)
    fps_hz = read_cih()

    images = io.ImageCollection(sorted(glob.glob('*.tif'), \
                                key=os.path.getmtime))

    time_ms = np.arange(end_frame-start_frame, dtype=float)
    centers = np.zeros((end_frame-start_frame,2), dtype=int)
    volume = np.zeros(end_frame-start_frame, dtype=int)

    save_folder = folder + '/info1'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, k in enumerate(range(start_frame, end_frame)):
    # for i, k in enumerate(range(49, end_frame)):
        print(k)
        image = images[k]
        image_cropped = image[ceiling:wall,:]
        image_filter = filters.median(image_cropped)
        image_binary1 = image_filter < threshold[0]
        image_binary2 = image_filter > threshold[1]
        image_binary = image_binary1 + image_binary2
        image_floodfill = binary_fill_holes(image_binary).astype(int)

        boundary = segmentation.find_boundaries(image_floodfill, connectivity=1, mode='outer', background=0)
        indices = np.transpose(np.where(boundary == 1))
        indices[:,[0,1]] = indices[:,[1,0]]
        indices = indices[indices[:, 1].argsort()]

        axis = np.mean(indices[:,0])
        vol = (np.pi/2)*np.trapz(np.power(indices[:,0]-axis,2))
        xx = axis
        yy = (np.pi/2)*np.trapz(indices[:,1]*np.power(indices[:,0]-axis,2))/vol

        # os.chdir(save_folder)
        #
        # plt.subplot(2,3,1)
        # plt.imshow(image_cropped,cmap='gray')
        # plt.title('Raw image' + str(k))
        # plt.axis('off')
        # plt.subplot(2,3,2)
        # plt.imshow(image_filter,cmap='gray')
        # plt.title('Median Filter image')
        # plt.axis('off')
        # plt.subplot(2,3,3)
        # plt.imshow(image_binary,cmap='gray')
        # plt.title('Binary image')
        # plt.axis('off')
        # plt.subplot(2,3,4)
        # plt.imshow(image_floodfill,cmap='gray')
        # plt.title('Floodfill image')
        # plt.axis('off')
        # plt.subplot(2,3,5)
        # plt.imshow(boundary,cmap='gray')
        # plt.title('Boundary image')
        # plt.axis('off')
        # plt.subplot(2,3,6)
        # plt.imshow(image_cropped,cmap='gray')
        # plt.scatter(indices[:,0], indices[:,1])
        # plt.scatter(xx, yy)
        # plt.title('Center image')
        # plt.axis('off')
        # # plt.savefig('image_cropped' + str(k) +'.png', bbox_inches='tight', format='png')
        # # plt.close()
        # plt.show()

        time_ms[i] = i*(1000.0/fps_hz)
        centers[i] = [xx, yy]
        volume[i] = vol

        # os.chdir(folder)

    os.chdir(save_folder)

    xc_mm = abs(centers[:,0]-centers[0,0])*(px_microns/1000.0)
    yc_mm = abs(centers[:,1]-(wall-ceiling+1))*(px_microns/1000.0)
    radius_mm = np.power((3*volume)/(4*np.pi),1/3)*(px_microns/1000)
    radius_correct_mm = radius_mm
    radius_correct_mm[yc_mm<1] = None

    index = np.argmin(np.abs(np.array(yc_mm)-1))+1

    if len(yc_mm[index:]) == 0:
        hr = 0
    else:
        hr = max(yc_mm[index:])

    # yc_correct_mm = yc_mm
    # yc_correct_mm[yc_mm<1] = None
    # time_correct_ms = time_ms
    # time_correct_ms[yc_mm<1] = None

    u_pre, xs, xi, x_fit_pre = horizontal_trajectory(t=time_ms[:index]-time_ms[index], r=xc_mm[:index])
    ys, vs, yi, vi, y_fit_pre, d = vertical_trajectory(t=time_ms[:index]-time_ms[index], r=yc_mm[:index])
    g = -9.81

    if hr >= 1:
        vr = np.power(2*abs(g)*(hr-1)/1000,1/2)
    else:
        vr = 0

    varepsilon = vr/(-vi)

    plt.figure(1, figsize=(15, 10))
    gs = gridspec.GridSpec(6, 5)

    plt.subplot(gs[0:2, 1:4])
    plt.scatter(time_ms,radius_mm, marker='.', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$R(t)$ $[mm]$')
    plt.title(r'$<R>$ = ' + str(round(np.mean(radius_mm),3)) + r' $mm$, ' \
               '$v_{i}$ = ' + str(round(vi,3)) + r' $m/s$, '\
               '$v_{r}$ = ' + str(round(vr,3)) + r' $m/s$ ')

    plt.subplot(gs[3:6, 0:2])
    plt.scatter(time_ms,xc_mm, marker='.', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$x(t)$ $[mm]$')
    plt.title(r'$x_{s} - x_{i}$ = ' + str(round(xs - xi,3)) + ' $mm$, ' \
               '$u$ $=$ ' + str(round(u_pre,3)) +' $m/s$ ')

    plt.subplot(gs[3:6, 3:5])
    plt.scatter(time_ms,yc_mm, marker='.', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$y(t)$ $[mm]$')
    plt.title(r'$y_{s} - y_{i}$ $=$ ' + str(round(ys-yi,3)) + ' $mm$, ' \
               '$v_{s}$ $=$ ' + str(round(vs,3)) + ' $m/s$, ' \
               '$h_{r}$ = ' + str(round(hr,3)) + ' $mm$ ')

    # plt.show()
    plt.savefig('input.pdf', format='pdf')

    txt_file = open("input.txt","w")

    txt_file.write(f"folder = {folder}\n")
    txt_file.write(f"drop = {drop}\n")
    txt_file.write(f"film = {film}\n")
    txt_file.write(f"hf_ideal_mum = {hf_ideal_mum}\n")
    txt_file.write(f"start_frame = {start_frame}\n")
    txt_file.write(f"end_frame = {end_frame}\n")
    txt_file.write(f"ceiling = {ceiling}\n")
    txt_file.write(f"wall = {wall}\n")
    txt_file.write(f"free_surface = {free_surface}\n")
    txt_file.write(f"threshold = {threshold[0], threshold[1]}\n")
    txt_file.write(f"px_microns = {px_microns}\n")
    txt_file.close()

    R = np.nanmean(radius_correct_mm)
    hf_ideal = hf_ideal_mum
    hf_real = (wall - free_surface)*px_microns
    rho_d, mu_d, gamma_d = liquid_properties(liquid = drop)
    rho_f, mu_f, gamma_f = liquid_properties(liquid = film)

    txt_file = open("params.txt","w")

    txt_file.write(f"DROP\n")
    txt_file.write(f"Radius = {R:.3f} mm\n")
    txt_file.write(f"Density = {rho_d:.0f} kg/m3\n")
    txt_file.write(f"Dynamic viscosity = {mu_d*1000:.3f} mPa.s\n")
    txt_file.write(f"Surface tension = {gamma_d*1000:.3f} mN/m\n")
    txt_file.write(f"Impact velocity = {abs(vi):.3f} m/s\n")
    txt_file.write(f"\n")
    txt_file.write(f"FILM\n")
    txt_file.write(f"Thickness_ideal = {hf_ideal} microns\n")
    txt_file.write(f"Thickness_real = {hf_real} microns\n")
    txt_file.write(f"Density = {rho_f:.0f} kg/m3\n")
    txt_file.write(f"Dynamic viscosity = {mu_f*1000:.3f} mPa.s\n")
    txt_file.write(f"Surface tension = {gamma_f*1000:.3f} mN/m\n")
    txt_file.write(f"\n")
    txt_file.write(f"OTHERS\n")
    txt_file.write(f"Gravity = 9.81 m/s2\n")
    txt_file.write(f"Surrounding air at STP conditions\n")
    txt_file.close()

    np.savetxt('We.txt', [(rho_d*((vi)**2)*R)/(1000*gamma_d)], fmt='%0.3f')
    np.savetxt('Bo.txt', [(rho_d*abs(g)*((R)**2))/(1000*1000*gamma_d)], fmt='%0.3f')
    np.savetxt('Oh.txt', [mu_d/np.power(rho_d*gamma_d*(R/1000),0.5)], fmt='%0.3f')
    np.savetxt('Dv.txt', [mu_f/np.power(rho_d*gamma_d*(R/1000),0.5)], fmt='%0.3f')
    np.savetxt('Hr_ideal.txt', [hf_ideal/(1000*R)], fmt='%0.3f')
    np.savetxt('Hr_real.txt', [hf_real/(1000*R)], fmt='%0.3f')
    np.savetxt('e.txt', [vr/(-vi)], fmt='%0.3f')

    return None

################################################################################

def horizontal_trajectory(t=[None], r=[None]):

    linear_params = np.polyfit(t, r, 1)
    u = linear_params[0]
    rs = linear_params[1] + (linear_params[0]*t[0])
    re = linear_params[1] + (linear_params[0]*t[-1])
    rfit = linear_params[1] + (linear_params[0]*t)

    return u, rs, re, rfit

################################################################################

def vertical_trajectory(t=[None], r=[None]):

    a = -9.81
    r_mod = r - ((a/(2.0*1000.0))*(t**2))
    linear_params = np.polyfit(t, r_mod, 1)
    rs = ((a/(2.0*1000.0))*(t[0]**2)) + \
         (linear_params[0]*(t[0]**1)) + \
         (linear_params[1]*(t[0]**0))
    vs = ((a/1000.0)*t[0]) + linear_params[0]

    if max(t)<=0:
        tt = t[-1]
    else:
        tt = -(linear_params[0]*1000.0)/a

    re = ((a/(2.0*1000.0))*(tt**2)) + \
         (linear_params[0]*(tt**1)) + \
         (linear_params[1]*(tt**0))
    ve = ((a/1000.0)*tt) + linear_params[0]
    rfit = ((a/(2.0*1000.0))*(t**2)) + \
           (linear_params[0]*(t**1)) + \
           (linear_params[1]*(t**0))
    d = tt - t[0]

    return rs, vs, re, ve, rfit, d

################################################################################

def liquid_properties(liquid = ''):

    if liquid == '1 cSt oil':
        rho = 816
        eta = 1*(rho/1000)*(1/1000)
        gamma = 0.017
    elif liquid == '10 cSt oil':
        rho = 930
        eta = 10*(rho/1000)*(1/1000)
        gamma = 0.020
    elif liquid == '20 cSt oil':
        rho = 950
        eta = 20*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == '35 cSt oil':
        rho = 960
        eta = 35*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == '50 cSt oil':
        rho = 960
        eta = 50*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == '100 cSt oil':
        rho = 960
        eta = 100*(rho/1000)*(1/1000)
        gamma = 0.021
    else:
        rho = None
        eta = None
        gamma = None

    return rho, eta, gamma

################################################################################

def read_cih():

    data = open(glob.glob('*.cih')[0],'r')

    rec = int(tuple(data)[15][19:])

    return rec

################################################################################
