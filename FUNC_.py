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
                    file            = '', \
                    drop            = '', \
                    film            = '', \
                    hf_ideal_mum    = None, \
                    before_impact   = [None,None], \
                    after_impact    = [None,None], \
                    vertical_limits = [None, None], \
                    free_surface    = None, \
                    threshold       = None, \
                    radius          = None, \
                    px_microns      = None):

    os.chdir(folder)

    fps_hz = int(tuple(open(file,'r'))[15][19:])
    images = io.ImageCollection(    sorted(glob.glob('*.tif'), \
                                    key=os.path.getmtime))
    n = len(images) - 1

    num1 = list(np.arange( before_impact[1] , before_impact[0] - 1 , -1))
    num2 = list(np.arange(  after_impact[0] ,  after_impact[1] + 1 , +1))

    num = list(np.sort(num1 + num2))

    m = len(num)
    o = len(num1)

    # plt.subplot(2,2,1)
    # plt.imshow(images[before_impact[0]],cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(images[before_impact[1]],cmap='gray')
    # plt.subplot(2,2,3)
    # plt.imshow(images[after_impact[0]],cmap='gray')
    # plt.subplot(2,2,4)
    # plt.imshow(images[after_impact[1]],cmap='gray')
    # plt.show()

    y_wall, y_top = vertical_limits[0], vertical_limits[1]

    x_center = int(np.shape(images[0])[1]/2)

    y_min, y_max = y_wall-360, y_wall+100
    x_min, x_max = x_center-150, x_center+150

    time_ms = np.arange(m, dtype=float)
    time_ref = before_impact[1]

    centers = np.zeros((m,2), dtype=int)
    volume = np.zeros(m, dtype=int)

    save_folder = folder + '/info'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, k in enumerate(num):

        print(k)

        # os.chdir(foldername)
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

        time_ms[i] = (k - time_ref)*(1000.0/fps_hz)
        centers[i] = [xx, yy]
        volume[i] = vol

        # os.chdir(save_folder)
        #
        # plt.subplot(2,3,1)
        # plt.imshow(image_cropped,cmap='gray')
        # plt.title('Raw image')
        # plt.axis('off')
        # plt.subplot(2,3,2)
        # plt.imshow(image_cropped_filter,cmap='gray')
        # plt.title('Median Filtered image')
        # plt.axis('off')
        # plt.subplot(2,3,3)
        # plt.imshow(binary,cmap='gray')
        # plt.title('Binary otsu image')
        # plt.axis('off')
        # plt.subplot(2,3,4)
        # plt.imshow(closing,cmap='gray')
        # plt.title('Closing image')
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
        # plt.savefig('image_cropped' + str(k) +'.png', bbox_inches='tight', format='png')
        # plt.close()
        # # plt.show()

    os.chdir(save_folder)

    xc_mm = abs(centers[:,0]-centers[0,0])*(px_microns/1000.0)
    yc_mm = abs(centers[:,1]-(y_wall-y_min+1))*(px_microns/1000.0)
    radius_mm = np.power((3*volume)/(4*np.pi),1/3)*(px_microns/1000)

    # before impact
    avg_radius1_mm = np.mean(radius_mm[:o])
    radius1_mm_fitted = np.ones(len(radius_mm[:o]))* avg_radius1_mm
    u_pre, xs, xi, xfit_pre = horizontal_trajectory(t=time_ms[:o], r=xc_mm[:o])
    ys, vs, yi, vi, yfit_pre, t_pre = vertical_trajectory(t=time_ms[:o], r=yc_mm[:o])

    plt.figure(1, figsize=(15, 10))
    gs = gridspec.GridSpec(6, 5)

    plt.subplot(gs[0:2, 1:4])
    plt.scatter(time_ms[:o],radius_mm[:o], marker='.', color='black')
    plt.plot(time_ms[:o],radius1_mm_fitted, linestyle='--', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$R(t)$ $[mm]$')
    plt.ylim(0.5,1.5)
    plt.title(r'$<R>$ = ' + str(round(avg_radius1_mm,3)) + r' $mm$, ' \
               '$v_{i}$ = ' + str(round(vi,3)) + r' $m/s$, '\
               '$y_{s} - y_{i}$ = ' + str(round(ys - yi,3)) + r' $mm$ ')

    plt.subplot(gs[3:6, 0:2])
    plt.scatter(time_ms[:o],xc_mm[:o], marker='.', color='black')
    plt.plot(time_ms[:o],xfit_pre, linestyle='--', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$x(t)$ $[mm]$')
    plt.title(r'$x_{s} - x_{i}$ = ' + str(round(xs - xi,3)) + ' $mm$, ' \
               '$u$ $=$ ' + str(round(u_pre,3)) +' $m/s$ ')

    plt.subplot(gs[3:6, 3:5])
    plt.scatter(time_ms[:o],yc_mm[:o], marker='.', color='black')
    plt.plot(time_ms[:o],yfit_pre, linestyle='--', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$y(t)$ $[mm]$')
    plt.title(r'$y_{i}$ $=$ ' + str(round(yi,3)) + ' $mm$, ' \
               '$v_{s}$ $=$ ' + str(round(vs,3)) + ' $m/s$, ' \
               '$t_{i} - t_{s}$ = ' + str(round(t_pre,3)) + ' $ms$ ')

    plt.savefig('before_impact.pdf', format='pdf')

    #after impact
    avg_radius2_mm = np.mean(radius_mm[o:])
    radius2_mm_fitted = np.ones(len(radius_mm[o:]))* avg_radius2_mm
    u_post, xr, xe, xfit_post = horizontal_trajectory(t=time_ms[o:], r=xc_mm[o:])
    yr, vr, ye, ve, yfit_post, t_post = vertical_trajectory(t=time_ms[o:], r=yc_mm[o:])

    plt.figure(2, figsize=(15, 10))
    gs = gridspec.GridSpec(6, 5)

    plt.subplot(gs[0:2, 1:4])
    plt.scatter(time_ms[o:],radius_mm[o:], marker='.', color='black')
    plt.plot(time_ms[o:],radius2_mm_fitted, linestyle='--', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$R(t)$ $[mm]$')
    plt.ylim(0.5,1.5)
    plt.title(r'$<R>$ = ' + str(round(avg_radius2_mm,3)) + r' $mm$, ' \
               '$v_{r}$ = ' + str(round(vr,3)) + r' $m/s$, '\
               '$y_{e} - y_{r}$ = ' + str(round(ye - yr,3)) + r' $mm$ ')

    plt.subplot(gs[3:6, 0:2])
    plt.scatter(time_ms[o:],xc_mm[o:], marker='.', color='black')
    plt.plot(time_ms[o:],xfit_post, linestyle='--', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$x(t)$ $[mm]$')
    plt.title(r'$x_{e} - x_{r}$ = ' + str(round(xe - xr,3)) + ' $mm$, ' \
               '$u$ $=$ ' + str(round(u_post,3)) +' $m/s$ ')

    plt.subplot(gs[3:6, 3:5])
    plt.scatter(time_ms[o:],yc_mm[o:], marker='.', color='black')
    plt.plot(time_ms[o:],yfit_post, linestyle='--', color='black')
    plt.xlabel('$t$ $[ms]$')
    plt.ylabel('$y(t)$ $[mm]$')
    plt.title(r'$y_{r}$ $=$ ' + str(round(yr,3)) + ' $mm$, ' \
               '$v_{e}$ $=$ ' + str(round(ve,3)) + ' $m/s$, ' \
               '$t_{e} - t_{r}$ = ' + str(round(t_post,3)) + ' $ms$ ')

    plt.savefig('after_impact.pdf', format='pdf')

    txt_file = open("impactspeed_sideview_info.txt","w")
    txt_file.write(f"1 pixel = {px_microns} microns\n")
    txt_file.write(f"Recording speed = {fps_hz} Hz\n")
    txt_file.write(f"Image number - Before impact = {before_impact[0], before_impact[1]}\n")
    txt_file.write(f"Image number -  After impact = {after_impact[0], after_impact[1]}\n")
    txt_file.write(f"Wall, Ceiling = {vertical_limits[0], vertical_limits[1]}\n")
    txt_file.write(f"free surface = {free_surface}\n")
    txt_file.write(f"threshold = {threshold}\n")
    txt_file.write(f"radius = {radius}\n")
    txt_file.close()

    R = (avg_radius1_mm + avg_radius2_mm)/2
    hf_ideal = hf_ideal_mum
    hf_real = (free_surface - y_wall)*px_microns
    g = -9.81
    rho_d, mu_d, gamma_d = liquid_properties(liquid = drop)
    rho_f, mu_f, gamma_f = liquid_properties(liquid = film)

    np.savetxt('nd_film_thickness_ideal.txt', [hf_ideal/(1000*R)], fmt='%0.3f')
    np.savetxt('nd_film_thickness_real.txt',  [hf_real/(1000*R)], fmt='%0.3f')
    np.savetxt('drop_We_R.txt', [(rho_d*((vi)**2)*R)/(1000*gamma_d)], fmt='%0.3f')
    np.savetxt('drop_Bo_R.txt', [(rho_d*abs(g)*((R)**2))/(1000*1000*gamma_d)], fmt='%0.3f')
    np.savetxt('drop_Oh_R.txt', [mu_d/np.power(rho_d*gamma_d*(R/1000),0.5)], fmt='%0.3f')
    np.savetxt('nd_drop_viscosity.txt', [mu_d/mu_f], fmt='%0.3f')
    np.savetxt('film_Oh_R.txt', [mu_f/np.power(rho_f*gamma_f*(R/1000),0.5)], fmt='%0.3f')
    np.savetxt('restitution_coefficient.txt', [vr/(-vi)], fmt='%0.3f')

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
