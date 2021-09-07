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
from scipy.ndimage import morphology
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt

import skimage
from skimage import io
from skimage import util
from skimage import color
from skimage import feature

from skimage import segmentation
from skimage.segmentation import active_contour
from skimage.segmentation import morphological_geodesic_active_contour

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

def info(dir=''):

    #lengthscale
    os.chdir(dir + r'\reference_lengthscale')
    px_microns = np.loadtxt(r'px_microns.txt')

    #background
    os.chdir(dir + r'\background')
    global_background = io.imread('background.tif')

    if np.min(global_background) == 0:
        global_background = global_background + 1

    return px_microns, global_background

################################################################################

def impact_info(\
                 folder     = '', \
                 px_microns = None, \
                 global_background = None):

    haha = [str.start() for str in re.finditer('_', folder)]
    drop = int(folder[haha[-9]+1:haha[-8]-3])                                   # kinematic viscosity of the drop in cSt
    film = int(folder[haha[-4]+1:haha[-3]-3])                                   # kinematic viscosity of the film in cSt
    hf_ideal_mum = int(folder[haha[-3]+1:haha[-2]-3])                           # thickness of the film in microns

    # hf_ideal_mum = int(folder[haha[-3]+1:haha[-2]-2])                           # thickness of the film in mm
    # hf_ideal_mum = hf_ideal_mum*1000

    haha1 = [str.start() for str in re.finditer('H_', folder)]
    haha2 = [str.start() for str in re.finditer('R_', folder)]
    needle_height = int(folder[haha1[-1]+2:haha2[0]])

    # print(drop, film, hf_ideal_mum)
    # input()

    # drop = 20
    # film = 1
    # hf_ideal_mum = 10
    # needle_height = 3

    os.chdir(folder)
    os.chdir('..')

    # background = np.loadtxt('background.txt')
    # background = io.imread('background.tif')

    # ceiling = int(np.loadtxt('ceiling.txt'))
    # interface = int(np.loadtxt('interface.txt'))
    start = int(np.loadtxt('start.txt'))
    # stop = int(np.loadtxt('stop.txt'))
    # threshold = np.loadtxt('threshold.txt')
    threshold1 = np.loadtxt('threshold.txt')[0]
    threshold2 = np.loadtxt('threshold.txt')[1]
    wall = int(np.loadtxt('wall.txt'))
    # background = np.loadtxt('background.txt')

    # background_start = int(np.loadtxt('background_params.txt')[0])
    # background_end   = int(np.loadtxt('background_params.txt')[1])
    # background_ratio = np.loadtxt('background_params.txt')[2]

    os.chdir(folder)
    os.chdir('input')

    ceiling = int(np.loadtxt('ceiling.txt'))
    interface = int(np.loadtxt('interface.txt'))
    # start = int(np.loadtxt('start.txt'))
    stop = int(np.loadtxt('stop.txt'))
    # threshold = np.loadtxt('threshold.txt')
    # threshold1 = np.loadtxt('threshold.txt')[0]
    # threshold2 = np.loadtxt('threshold.txt')[1]
    # wall = int(np.loadtxt('wall.txt'))
    # background = np.loadtxt('background.txt')

    impact_frame = int(np.loadtxt('critical_frames.txt')[0])
    rebound_frame = int(np.loadtxt('critical_frames.txt')[1])
    reimpact_frame = np.loadtxt('critical_frames.txt')[2]

    background_start = int(np.loadtxt('background_params.txt')[0])
    background_end   = int(np.loadtxt('background_params.txt')[1])
    background_ratio = np.loadtxt('background_params.txt')[2]

    median_filter_size = 3

    # if interface > wall:
    #     interface = wall

    # wall = interface

    os.chdir(folder)

    fps_hz, color_bit, width, height = read_cih()

    images = io.ImageCollection(sorted(glob.glob('*.tif'), \
                                key=os.path.getmtime))

    time_ms = np.zeros(stop-start, dtype=float)
    xc_mm = np.zeros(stop-start, dtype=float)
    yc_mm = np.zeros(stop-start, dtype=float)
    radius_mm = np.zeros(stop-start, dtype=float)
    area_mm2 = np.zeros(stop-start, dtype=float)
    spread_mm = np.zeros(stop-start, dtype=float)
    y_bottom_drop_time = np.zeros(stop-start, dtype=float)
    # xc_px = np.zeros(stop-start, dtype=float)
    # yc_px = np.zeros(stop-start, dtype=float)
    # fr = np.zeros(stop-start, dtype=float)

    save_folder = folder + '/output'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_save_folder = folder + '/image_processing'
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

    phase = 'fall'
    impact_index = None
    bounce_index = None
    cyclic_index = None

    if needle_height <= 4:
        background_up = global_background[ceiling:interface,:]
        background_up[0:int(background_ratio*(interface-ceiling)),:] = 0
    else:
        background_up = images[background_start][ceiling:interface,:]
        background_up[0:int(background_ratio*(interface-ceiling)),:] = 0

    background_down = images[background_end][ceiling:interface,:]
    background_down[int(background_ratio*(interface-ceiling)):(interface-ceiling),:] = 0

    background = background_up + background_down

    if np.min(background) == 0:
        background = background + 1

    # plt.imshow(background, cmap='gray')
    # plt.show()

    os.chdir('output')

    plt.imshow(background, cmap='gray')
    plt.savefig('background.png', format='png')
    plt.close()

    os.chdir(folder)

    for i, k in enumerate(range(start, stop)):
    # for i, k in enumerate(range(84, stop)):
    # for i, k in enumerate([25, 332, 334, 336]):
        print(k, phase)
        image = images[k]

        image_cropped = image[ceiling:interface,:]
        image_divided  = np.divide(image_cropped, background)
        image_filtered = median_filter(input = image_divided, \
                                       size  = median_filter_size)

        if k<impact_frame:
            image_binary = image_filtered < threshold1
        elif k>rebound_frame:
            image_binary = image_filtered < threshold2
        else:
            image_binary = image_filtered < (threshold1 + threshold2)/2.0

        if any(image_binary[-1,:]):
            image_binary[-1,:] = binary_fill_holes(input = image_binary[-1,:])

        image_floodfill = binary_fill_holes(\
                                             input     = image_binary, \
                                             structure = None, \
                                             output    = None, \
                                             origin    = 0)

        if k>rebound_frame:
            image_floodfill[interface-ceiling-10:,:] = 0

        edges = image_floodfill ^ morphology.binary_dilation(image_floodfill)

        indices = np.transpose(np.where(edges == 1))
        indices[:,[0,1]] = indices[:,[1,0]]
        indices = indices[indices[:, 1].argsort()]

        if indices.size == 0:
            # indices[:,0] = (wall-interface)*np.ones(488)
            # indices[:,1] = (wall-interface)*np.ones(488)
            # indices = np.empty((10, 10))
            y_bottom_drop = 1
            axis = 1
            vol = 1
            surf = 1
            xx = 1
            yy = 1
            maxx = 1
            # maxx = 0
        else:
            y_bottom_drop = max(indices[:,1])
            axis = np.mean(indices[:,0])
            vol = (np.pi/2)*np.trapz(np.power(indices[:,0]-axis,2))
            surf = (2*np.pi)*np.trapz(np.power(indices[:,0]-axis,1))
            xx = axis
            yy = (np.pi/2)*np.trapz(indices[:,1]*np.power(indices[:,0]-axis,2))/vol
            maxx = (max(indices[:,0]) - min(indices[:,0]))/2
            # maxx = 0

        if i % 10 == 0:
            os.chdir(image_save_folder)

            plt.subplot(2,2,1)
            plt.imshow(image_cropped,cmap='gray')
            plt.title('Cropped image ' + str(k))
            plt.axis('off')
            plt.subplot(2,2,2)
            plt.imshow(image_binary,cmap='gray')
            plt.title('Binary image ' + str(k))
            plt.axis('off')
            plt.subplot(2,2,3)
            plt.imshow(image_floodfill,cmap='gray')
            plt.title('Floodfill image ' + str(k))
            plt.axis('off')
            # plt.subplot(2,2,3)
            # plt.imshow(image_closing,cmap='gray')
            # plt.title('Closing image ' + str(k))
            # plt.axis('off')
            # plt.subplot(2,2,3)
            # plt.imshow(boundary,cmap='gray')
            # plt.title('Boundary image ' + str(k))
            # plt.axis('off')
            plt.subplot(2,2,4)
            plt.imshow(image_filtered,cmap='gray')
            plt.scatter(indices[:,0], indices[:,1], marker='.')
            plt.scatter(xx, yy)
            plt.title('Full image ' + str(k))
            plt.axis('off')
            plt.savefig('image ' + str(k) +'.png', format='png')
            plt.close()
            # plt.show()

        # frames[i] = k
        # centers[i] = [xx, yy]
        # volume[i] = int(vol)

        time_ms[i] = (k-start)*(1000.0/fps_hz)
        xc_mm[i] = int((width/2)-xx)*(px_microns/1000.0)
        yc_mm[i] = int((interface-ceiling)-yy)*(px_microns/1000.0)
        radius_mm[i] = np.power((3*int(vol))/(4*np.pi),1/3)*(px_microns/1000.0)
        area_mm2[i] = surf*np.power((px_microns/1000.0),2)
        spread_mm[i] = maxx*(px_microns/1000.0)
        y_bottom_drop_time[i] = int((interface-ceiling)-y_bottom_drop)
        # xc_px[i] = xx
        # yc_px[i] = yy
        # fr[i] = i

        # os.chdir(image_save_folder)
        # np.savetxt(f'x_{k:03d}.txt', indices[:,0], fmt='%d')
        # np.savetxt(f'y_{k:03d}.txt', indices[:,1], fmt='%d')
        # np.savetxt(f'xc_{k:03d}.txt', [xx], fmt='%d')
        # np.savetxt(f'yc_{k:03d}.txt', [yy], fmt='%d')
        # np.savetxt(f'{k:03d}.txt', [k], fmt='%d')

        # if phase == 'fall':
        #     if yc_mm[i] < 1:
        #         impact_index = i
        #         impact_image = k
        #         phase = 'lubrication'
        # elif phase == 'lubrication':
        #     if yc_mm[i] > 1:
        #         bounce_index = i
        #         bounce_image = k
        #         phase = 'rise'
        # elif phase == 'rise':
        #     # if yc_mm[i] < 1:
        #     if k == stop:
        #         cyclic_index = i
        #         cyclic_image = k
        #         phase = 'cyclic'
        # elif phase == 'cyclic':
        #     break

        # if phase == 'fall':
        #     if y_bottom_drop_time[i] < 2:
        #         impact_index = i
        #         impact_image = k
        #         phase = 'lubrication'
        # elif phase == 'lubrication':
        #     if y_bottom_drop_time[i] > 5:
        #         bounce_index = i
        #         bounce_image = k
        #         phase = 'rise'
        # elif phase == 'rise':
        #     if y_bottom_drop_time[i] < 2:
        #         cyclic_index = i
        #         cyclic_image = k
        #         phase = 'cyclic'
        # elif phase == 'cyclic':
        #     break

        if phase == 'fall':
            if k == impact_frame:
                impact_index = i
                impact_image = k
                phase = 'lubrication'
        elif phase == 'lubrication':
            if k == rebound_frame:
                bounce_index = i
                bounce_image = k
                phase = 'rise'
        elif phase == 'rise':
            if k == reimpact_frame:
                cyclic_index = i
                cyclic_image = k
                phase = 'cyclic'
        elif phase == 'cyclic':
            break

        os.chdir(folder)

    os.chdir(save_folder)
    g = -9.81

    fig1 = plt.figure(1, figsize=(15, 10))

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,2,3)
    ax3 = plt.subplot(2,2,4)

    ax1.set_xlabel('$t$ $[ms]$')
    ax1.set_ylabel('$R(t)$ $[mm]$')

    ax2.set_xlabel('$t$ $[ms]$')
    ax2.set_ylabel('$x(t)$ $[mm]$')

    ax3.set_xlabel('$t$ $[ms]$')
    ax3.set_ylabel('$y(t)$ $[mm]$')

    ax1.scatter(time_ms[:i]-time_ms[impact_index], radius_mm[:i], marker='.', color='black')
    ax2.scatter(time_ms[:i]-time_ms[impact_index], xc_mm[:i], marker='.', color='black')
    ax3.scatter(time_ms[:i]-time_ms[impact_index], yc_mm[:i], marker='.', color='black')

    fig1.savefig('total.pdf', format='pdf')

    # SAVING yc_mm AND t_ms
    # np.savetxt('yc_mm.txt', yc_mm[:i], fmt='%0.3f')
    # np.savetxt('t_ms.txt', time_ms[:i], fmt='%0.3f')
    # np.savetxt('t_impact_ms.txt', [time_ms[impact_index]], fmt='%0.3f')
    # np.savetxt('t_bounce_ms.txt', [time_ms[bounce_index]], fmt='%0.3f')
    # np.savetxt('t_reimpact_ms.txt', [time_ms[cyclic_index]], fmt='%0.3f')

    #Impact
    # time_impact_ms = time_ms[:impact_index]-time_ms[impact_index]
    # xc_impact_mm = xc_mm[:impact_index]
    # yc_impact_mm = yc_mm[:impact_index]
    # radius_impact_mm = radius_mm[:impact_index]

    yi = 1

    # if needle_height <= 4:
    #     y_cutoff = yi*(1+0.05)
    #     index_cutoff = np.min(np.where(yc_mm < y_cutoff))
    # else:
    #     y_cutoff = yc_mm[0]
    #     index_cutoff = 0

    y_cutoff = yc_mm[0]
    index_cutoff = 0

    time_impact_ms = time_ms[index_cutoff:impact_index]-time_ms[impact_index]
    xc_impact_mm = xc_mm[index_cutoff:impact_index]
    yc_impact_mm = yc_mm[index_cutoff:impact_index]
    radius_impact_mm = radius_mm[index_cutoff:impact_index]

    if needle_height <= 4:
        u_pre, xs, xi, x_fit_pre = horizontal_trajectory(t=time_impact_ms, r=xc_impact_mm)
        ys, vs, yim, vim, y_fit_pre, delta_time_pre, vi, a = vertical_trajectory1(t=time_impact_ms, r=yc_impact_mm)
        g1 = a
    else:
        u_pre, xs, xi, x_fit_pre = horizontal_trajectory(t=time_impact_ms, r=xc_impact_mm)
        ys, vs, yim, vim, y_fit_pre, delta_time_pre, vi = vertical_trajectory(t=time_impact_ms, r=yc_impact_mm,a=g)
        g1 = g

    fig2 = plt.figure(2, figsize=(15, 10))

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,2,3)
    ax3 = plt.subplot(2,2,4)

    ax1.set_xlabel('$t$ $[ms]$')
    ax1.set_ylabel('$R(t)$ $[mm]$')

    ax2.set_xlabel('$t$ $[ms]$')
    ax2.set_ylabel('$x(t)$ $[mm]$')

    ax3.set_xlabel('$t$ $[ms]$')
    ax3.set_ylabel('$y(t)$ $[mm]$')

    ax1.scatter(time_impact_ms, radius_impact_mm, marker='.', color='black')
    ax1.plot(time_impact_ms, np.ones(len(time_impact_ms))*np.mean(radius_impact_mm), linestyle='--', color='red')
    ax1.set_title(r'$<R>$ = ' + str(round(np.mean(radius_impact_mm),3)) + r' $mm$, ' \
               '$v_{i}$ = ' + str(round(vi,3)) + r' $m/s$, '\
               '$t_{i}$ = ' + str(round(delta_time_pre,3)) + r' $ms$ ')

    ax2.scatter(time_impact_ms, xc_impact_mm, marker='.', color='black')
    ax2.plot(time_impact_ms, x_fit_pre, linestyle='--', color='red')
    ax2.set_title(r'$x_{s} - x_{i}$ = ' + str(round(xs - xi,3)) + ' $mm$, ' \
               '$u$ $=$ ' + str(round(u_pre,3)) +' $m/s$ ')

    ax3.scatter(time_impact_ms, yc_impact_mm, marker='.', color='black')
    ax3.plot(time_impact_ms, y_fit_pre, linestyle='--', color='red')
    ax3.set_title(r'$y_{s} - y_{i}$ $=$ ' + str(round(ys-yi,3)) + ' $mm$, ' \
               '$v_{s}$ $=$ ' + str(round(vs,3)) + ' $m/s$, ' \
               '$g$ = ' + str(round(g1,3)) + ' $m/s^2$ ')

    fig2.savefig('impact.pdf', format='pdf')

    if cyclic_index != None and cyclic_index-bounce_index > 10:
        #Bounce
        time_bounce_ms = time_ms[bounce_index:cyclic_index]-time_ms[impact_index]
        xc_bounce_mm = xc_mm[bounce_index:cyclic_index]
        yc_bounce_mm = yc_mm[bounce_index:cyclic_index]
        radius_bounce_mm = radius_mm[bounce_index:cyclic_index]
        bounce_height_mm = max(yc_bounce_mm)
        u_post, xr, xe, x_fit_post = horizontal_trajectory(t=time_bounce_ms, r=xc_bounce_mm)
        yre, vre, ye, ve, y_fit_post, delta_time_post, vr = vertical_trajectory(t=time_bounce_ms, r=yc_bounce_mm,a=g)

        yr = 1

        fig3 = plt.figure(3, figsize=(15, 10))

        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,2,3)
        ax3 = plt.subplot(2,2,4)

        ax1.set_xlabel('$t$ $[ms]$')
        ax1.set_ylabel('$R(t)$ $[mm]$')

        ax2.set_xlabel('$t$ $[ms]$')
        ax2.set_ylabel('$x(t)$ $[mm]$')

        ax3.set_xlabel('$t$ $[ms]$')
        ax3.set_ylabel('$y(t)$ $[mm]$')

        ax1.scatter(time_bounce_ms, radius_bounce_mm, marker='.', color='black')
        ax1.plot(time_bounce_ms, np.ones(len(time_bounce_ms))*np.mean(radius_bounce_mm), linestyle='--', color='red')
        ax1.set_title(r'$<R>$ = ' + str(round(np.mean(radius_bounce_mm),3)) + r' $mm$, ' \
                   '$v_{r}$ = ' + str(round(vr,3)) + r' $m/s$, '\
                   '$t_{r}$ = ' + str(round(delta_time_post,3)) + r' $ms$ ')


        ax2.scatter(time_bounce_ms, xc_bounce_mm, marker='.', color='black')
        ax2.plot(time_bounce_ms, x_fit_post, linestyle='--', color='red')
        ax2.set_title(r'$x_{e} - x_{r}$ = ' + str(round(xe - xr,3)) + ' $mm$, ' \
                   '$u$ $=$ ' + str(round(u_post,3)) +' $m/s$ ')


        ax3.scatter(time_bounce_ms, yc_bounce_mm, marker='.', color='black')
        ax3.plot(time_bounce_ms, y_fit_post, linestyle='--', color='red')
        ax3.set_title(r'$y_{e} - y_{r}$ $=$ ' + str(round(ye-yr,3)) + ' $mm$, ' \
                   '$v_{e}$ $=$ ' + str(round(ve,3)) + ' $m/s$, ' \
                   '$g$ = ' + str(round(g,3)) + ' $m/s^2$ ')

        fig3.savefig('bounce.pdf', format='pdf')

    else:
        vr = 0

    txt_file = open("input.txt","w")

    txt_file.write(f"folder = {folder}\n")
    txt_file.write(f"start = {start}\n")
    txt_file.write(f"stop = {stop}\n")
    txt_file.write(f"ceiling = {ceiling}\n")
    txt_file.write(f"interface = {interface}\n")
    txt_file.write(f"wall = {wall}\n")
    txt_file.write(f"threshold = {threshold1} {threshold2}\n")
    txt_file.write(f"px_microns = {px_microns}\n")
    txt_file.close()

    R = np.nanmean(radius_mm[:impact_index])
    # R = np.nanmean(radius_mm[yc_mm>1])
    hf_ideal = hf_ideal_mum
    hf_real = (wall - interface)*px_microns
    rho_d, mu_d, gamma_d = liquid_properties(liquid = drop)
    rho_f, mu_f, gamma_f = liquid_properties(liquid = film)

    txt_file = open("params.txt","w")

    txt_file.write(f"DROP\n")
    txt_file.write(f"Radius = {R:.3f} mm\n")
    txt_file.write(f"Density = {rho_d:.0f} kg/m3\n")
    txt_file.write(f"Dynamic viscosity = {mu_d*1000:.3f} mPa.s\n")
    txt_file.write(f"Surface tension = {gamma_d*1000:.3f} mN/m\n")
    txt_file.write(f"Impact velocity = {abs(vi):.3f} m/s\n")
    txt_file.write(f"Bounce height = {bounce_height_mm:.3f} mm\n")
    txt_file.write(f"Contact time = {(rebound_frame-impact_frame)*0.1:.3f} ms\n")
    txt_file.write(f"Maximum spread = {max(spread_mm):.3f} mm\n")
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

    # np.savetxt('fr.txt', fr, fmt='%d')
    # np.savetxt('yc_px.txt', yc_px, fmt='%d')
    # np.savetxt('xc_px.txt', xc_px, fmt='%d')

    if bounce_index != None:
        np.savetxt('Tt.txt', [(bounce_image-impact_image)*(1/fps_hz)/np.sqrt((rho_d*((R/1000)**3))/(gamma_d))], fmt='%0.3f')
        np.savetxt('Sr.txt', [max(spread_mm)/R], fmt='%0.3f')
        np.savetxt('e.txt', [vr/(-vi)], fmt='%0.3f')
        np.savetxt('e_bh.txt', [np.sqrt(2*abs(g)*(bounce_height_mm-R)*(10**-3))/(-vi)], fmt='%0.3f')
    else:
        np.savetxt('Tt.txt', [0], fmt='%0.3f')
        np.savetxt('Sr.txt', [0], fmt='%0.3f')
        np.savetxt('e.txt', [0], fmt='%0.3f')

    return None

################################################################################

def inviscid_pool_info(\
                 folder     = '', \
                 px_microns = None, \
                 global_background = None):

    haha = [str.start() for str in re.finditer('_', folder)]
    drop = int(folder[haha[-9]+1:haha[-8]-3])                                   # kinematic viscosity of the drop in cSt
    film = int(folder[haha[-4]+1:haha[-3]-3])                                   # kinematic viscosity of the film in cSt
    # hf_ideal_mum = int(folder[haha[-3]+1:haha[-2]-3])                           # thickness of the film in microns

    hf_ideal_mum = int(folder[haha[-3]+1:haha[-2]-2])                           # thickness of the film in mm
    hf_ideal_mum = hf_ideal_mum*1000

    haha1 = [str.start() for str in re.finditer('H_', folder)]
    haha2 = [str.start() for str in re.finditer('R_', folder)]
    needle_height = int(folder[haha1[-1]+2:haha2[0]])

    # print(drop, film, hf_ideal_mum)
    # input()

    os.chdir(folder)
    os.chdir('..')

    # background = np.loadtxt('background.txt')
    # background = io.imread('background.tif')

    # ceiling = int(np.loadtxt('ceiling.txt'))
    # interface = int(np.loadtxt('interface.txt'))
    start = int(np.loadtxt('start.txt'))
    stop = int(np.loadtxt('stop.txt'))
    # threshold = np.loadtxt('threshold.txt')
    threshold1 = np.loadtxt('threshold.txt')[0]
    threshold2 = np.loadtxt('threshold.txt')[1]
    wall = int(np.loadtxt('wall.txt'))
    # background = np.loadtxt('background.txt')

    # background_start = int(np.loadtxt('background_params.txt')[0])
    # background_end   = int(np.loadtxt('background_params.txt')[1])
    # background_ratio = np.loadtxt('background_params.txt')[2]

    os.chdir(folder)
    os.chdir('input')

    ceiling = int(np.loadtxt('ceiling.txt'))
    interface = int(np.loadtxt('interface.txt'))
    # start = int(np.loadtxt('start.txt'))
    # stop = int(np.loadtxt('stop.txt'))
    # threshold = np.loadtxt('threshold.txt')
    # threshold1 = np.loadtxt('threshold.txt')[0]
    # threshold2 = np.loadtxt('threshold.txt')[1]
    # wall = int(np.loadtxt('wall.txt'))
    # background = np.loadtxt('background.txt')

    impact_frame = int(np.loadtxt('critical_frames.txt')[0])
    rebound_frame = int(np.loadtxt('critical_frames.txt')[1])
    reimpact_frame = np.loadtxt('critical_frames.txt')[2]

    background_start = int(np.loadtxt('background_params.txt')[0])
    background_end   = int(np.loadtxt('background_params.txt')[1])
    background_ratio = np.loadtxt('background_params.txt')[2]

    median_filter_size = 3

    # if interface > wall:
    #     interface = wall

    # wall = interface

    os.chdir(folder)

    fps_hz, color_bit, width, height = read_cih()

    images = io.ImageCollection(sorted(glob.glob('*.tif'), \
                                key=os.path.getmtime))

    time_ms = np.zeros(stop-start, dtype=float)
    xc_mm = np.zeros(stop-start, dtype=float)
    yc_mm = np.zeros(stop-start, dtype=float)
    radius_mm = np.zeros(stop-start, dtype=float)
    area_mm2 = np.zeros(stop-start, dtype=float)
    spread_mm = np.zeros(stop-start, dtype=float)
    y_bottom_drop_time = np.zeros(stop-start, dtype=float)

    save_folder = folder + '/output'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_save_folder = folder + '/image_processing'
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

    phase = 'fall'
    impact_index = None
    bounce_index = None
    cyclic_index = None

    if needle_height <= 4:
        background_up = global_background[ceiling:interface,:]
        background_up[0:int(background_ratio*(interface-ceiling)),:] = 0
    else:
        background_up = images[background_start][ceiling:interface,:]
        background_up[0:int(background_ratio*(interface-ceiling)),:] = 0

    background_down = images[background_end][ceiling:interface,:]
    background_down[int(background_ratio*(interface-ceiling)):(interface-ceiling),:] = 0

    background = background_up + background_down

    if np.min(background) == 0:
        background = background + 1

    # plt.imshow(background, cmap='gray')
    # plt.show()

    os.chdir('output')

    plt.imshow(background, cmap='gray')
    plt.savefig('background.png', format='png')
    plt.close()

    os.chdir(folder)

    for i, k in enumerate(range(start, stop)):
    # for i, k in enumerate(range(671, stop)):
        print(k, phase)
        image = images[k]

        image_cropped = image[ceiling:interface,:]
        image_divided  = np.divide(image_cropped, background)
        image_filtered = median_filter(input = image_divided, \
                                       size  = median_filter_size)

        if k<impact_frame:
            image_binary = image_filtered < threshold1
        elif k>rebound_frame:
            image_binary = image_filtered < threshold2
        else:
            image_binary = image_filtered < (threshold1 + threshold2)/2.0

        if any(image_binary[-1,:]):
            image_binary[-1,:] = binary_fill_holes(input = image_binary[-1,:])

        image_floodfill = binary_fill_holes(\
                                             input     = image_binary, \
                                             structure = None, \
                                             output    = None, \
                                             origin    = 0)

        if k>rebound_frame:
            image_floodfill[interface-ceiling-20:,:] = 0

        edges = image_floodfill ^ morphology.binary_dilation(image_floodfill)

        indices = np.transpose(np.where(edges == 1))
        indices[:,[0,1]] = indices[:,[1,0]]
        indices = indices[indices[:, 1].argsort()]

        if indices.size == 0:
            # indices[:,0] = (wall-interface)*np.ones(488)
            # indices[:,1] = (wall-interface)*np.ones(488)
            # indices = np.empty((10, 10))
            y_bottom_drop = 1
            axis = 1
            vol = 1
            surf = 1
            xx = 1
            yy = 1
            maxx = 1
            # maxx = 0
        else:
            y_bottom_drop = max(indices[:,1])
            axis = np.mean(indices[:,0])
            vol = (np.pi/2)*np.trapz(np.power(indices[:,0]-axis,2))
            surf = (2*np.pi)*np.trapz(np.power(indices[:,0]-axis,1))
            xx = axis
            yy = (np.pi/2)*np.trapz(indices[:,1]*np.power(indices[:,0]-axis,2))/vol
            maxx = (max(indices[:,0]) - min(indices[:,0]))/2
            # maxx = 0


        if i % 10 == 0:
            os.chdir(image_save_folder)

            plt.subplot(2,2,1)
            plt.imshow(image_cropped,cmap='gray')
            plt.title('Cropped image ' + str(k))
            plt.axis('off')
            plt.subplot(2,2,2)
            plt.imshow(image_binary,cmap='gray')
            plt.title('Binary image ' + str(k))
            plt.axis('off')
            plt.subplot(2,2,3)
            plt.imshow(image_floodfill,cmap='gray')
            plt.title('Floodfill image ' + str(k))
            plt.axis('off')
            # plt.subplot(2,2,3)
            # plt.imshow(image_closing,cmap='gray')
            # plt.title('Closing image ' + str(k))
            # plt.axis('off')
            # plt.subplot(2,2,3)
            # plt.imshow(boundary,cmap='gray')
            # plt.title('Boundary image ' + str(k))
            # plt.axis('off')
            plt.subplot(2,2,4)
            plt.imshow(image_filtered,cmap='gray')
            plt.scatter(indices[:,0], indices[:,1], marker='.')
            plt.scatter(xx, yy)
            plt.title('Full image ' + str(k))
            plt.axis('off')
            plt.savefig('image ' + str(k) +'.png', format='png')
            plt.close()
            # plt.show()

        # frames[i] = k
        # centers[i] = [xx, yy]
        # volume[i] = int(vol)

        time_ms[i] = (k-start)*(1000.0/fps_hz)
        xc_mm[i] = int((width/2)-xx)*(px_microns/1000.0)
        yc_mm[i] = int((interface-ceiling)-yy)*(px_microns/1000.0)
        radius_mm[i] = np.power((3*int(vol))/(4*np.pi),1/3)*(px_microns/1000.0)
        area_mm2[i] = surf*np.power((px_microns/1000.0),2)
        spread_mm[i] = maxx*(px_microns/1000.0)
        y_bottom_drop_time[i] = int((interface-ceiling)-y_bottom_drop)

        # if phase == 'fall':
        #     if yc_mm[i] < 1:
        #         impact_index = i
        #         impact_image = k
        #         phase = 'lubrication'
        # elif phase == 'lubrication':
        #     if yc_mm[i] > 1:
        #         bounce_index = i
        #         bounce_image = k
        #         phase = 'rise'
        # elif phase == 'rise':
        #     # if yc_mm[i] < 1:
        #     if k == stop:
        #         cyclic_index = i
        #         cyclic_image = k
        #         phase = 'cyclic'
        # elif phase == 'cyclic':
        #     break

        # if phase == 'fall':
        #     if y_bottom_drop_time[i] < 2:
        #         impact_index = i
        #         impact_image = k
        #         phase = 'lubrication'
        # elif phase == 'lubrication':
        #     if y_bottom_drop_time[i] > 10:
        #         bounce_index = i
        #         bounce_image = k
        #         phase = 'rise'
        # elif phase == 'rise':
        #     if y_bottom_drop_time[i] < 2:
        #         cyclic_index = i
        #         cyclic_image = k
        #         phase = 'cyclic'
        # elif phase == 'cyclic':
        #     break

        if phase == 'fall':
            if k == impact_frame:
                impact_index = i
                impact_image = k
                phase = 'lubrication'
        elif phase == 'lubrication':
            if k == rebound_frame:
                bounce_index = i
                bounce_image = k
                phase = 'rise'
        elif phase == 'rise':
            if k == reimpact_frame:
                cyclic_index = i
                cyclic_image = k
                phase = 'cyclic'
        elif phase == 'cyclic':
            break

        os.chdir(folder)

    os.chdir(save_folder)
    g = -9.81

    fig1 = plt.figure(1, figsize=(15, 10))

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,2,3)
    ax3 = plt.subplot(2,2,4)

    ax1.set_xlabel('$t$ $[ms]$')
    ax1.set_ylabel('$R(t)$ $[mm]$')

    ax2.set_xlabel('$t$ $[ms]$')
    ax2.set_ylabel('$x(t)$ $[mm]$')

    ax3.set_xlabel('$t$ $[ms]$')
    ax3.set_ylabel('$y(t)$ $[mm]$')

    ax1.scatter(time_ms[:i]-time_ms[impact_index], radius_mm[:i], marker='.', color='black')
    ax2.scatter(time_ms[:i]-time_ms[impact_index], xc_mm[:i], marker='.', color='black')
    ax3.scatter(time_ms[:i]-time_ms[impact_index], yc_mm[:i], marker='.', color='black')

    fig1.savefig('total.pdf', format='pdf')

    #Impact
    # time_impact_ms = time_ms[:impact_index]-time_ms[impact_index]
    # xc_impact_mm = xc_mm[:impact_index]
    # yc_impact_mm = yc_mm[:impact_index]
    # radius_impact_mm = radius_mm[:impact_index]

    yi = 1

    # if needle_height <= 4:
    #     y_cutoff = yi*(1+0.05)
    #     index_cutoff = np.min(np.where(yc_mm < y_cutoff))
    # else:
    #     y_cutoff = yc_mm[0]
    #     index_cutoff = 0

    y_cutoff = yc_mm[0]
    index_cutoff = 0

    time_impact_ms = time_ms[index_cutoff:impact_index]-time_ms[impact_index]
    xc_impact_mm = xc_mm[index_cutoff:impact_index]
    yc_impact_mm = yc_mm[index_cutoff:impact_index]
    radius_impact_mm = radius_mm[index_cutoff:impact_index]

    if needle_height <= 4:
        u_pre, xs, xi, x_fit_pre = horizontal_trajectory(t=time_impact_ms, r=xc_impact_mm)
        ys, vs, yim, vim, y_fit_pre, delta_time_pre, vi, a = vertical_trajectory1(t=time_impact_ms, r=yc_impact_mm)
        g1 = a
    else:
        u_pre, xs, xi, x_fit_pre = horizontal_trajectory(t=time_impact_ms, r=xc_impact_mm)
        ys, vs, yim, vim, y_fit_pre, delta_time_pre, vi = vertical_trajectory(t=time_impact_ms, r=yc_impact_mm,a=g)
        g1 = g

    fig2 = plt.figure(2, figsize=(15, 10))

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,2,3)
    ax3 = plt.subplot(2,2,4)

    ax1.set_xlabel('$t$ $[ms]$')
    ax1.set_ylabel('$R(t)$ $[mm]$')

    ax2.set_xlabel('$t$ $[ms]$')
    ax2.set_ylabel('$x(t)$ $[mm]$')

    ax3.set_xlabel('$t$ $[ms]$')
    ax3.set_ylabel('$y(t)$ $[mm]$')

    ax1.scatter(time_impact_ms, radius_impact_mm, marker='.', color='black')
    ax1.plot(time_impact_ms, np.ones(len(time_impact_ms))*np.mean(radius_impact_mm), linestyle='--', color='red')
    ax1.set_title(r'$<R>$ = ' + str(round(np.mean(radius_impact_mm),3)) + r' $mm$, ' \
               '$v_{i}$ = ' + str(round(vi,3)) + r' $m/s$, '\
               '$t_{i}$ = ' + str(round(delta_time_pre,3)) + r' $ms$ ')

    ax2.scatter(time_impact_ms, xc_impact_mm, marker='.', color='black')
    ax2.plot(time_impact_ms, x_fit_pre, linestyle='--', color='red')
    ax2.set_title(r'$x_{s} - x_{i}$ = ' + str(round(xs - xi,3)) + ' $mm$, ' \
               '$u$ $=$ ' + str(round(u_pre,3)) +' $m/s$ ')

    ax3.scatter(time_impact_ms, yc_impact_mm, marker='.', color='black')
    ax3.plot(time_impact_ms, y_fit_pre, linestyle='--', color='red')
    ax3.set_title(r'$y_{s} - y_{i}$ $=$ ' + str(round(ys-yi,3)) + ' $mm$, ' \
               '$v_{s}$ $=$ ' + str(round(vs,3)) + ' $m/s$, ' \
               '$g$ = ' + str(round(g1,3)) + ' $m/s^2$ ')

    fig2.savefig('impact.pdf', format='pdf')

    if cyclic_index != None and cyclic_index-bounce_index > 10:
        #Bounce
        time_bounce_ms = time_ms[bounce_index:cyclic_index]-time_ms[impact_index]
        xc_bounce_mm = xc_mm[bounce_index:cyclic_index]
        yc_bounce_mm = yc_mm[bounce_index:cyclic_index]
        radius_bounce_mm = radius_mm[bounce_index:cyclic_index]
        bounce_height_mm = max(yc_bounce_mm)
        u_post, xr, xe, x_fit_post = horizontal_trajectory(t=time_bounce_ms, r=xc_bounce_mm)
        yre, vre, ye, ve, y_fit_post, delta_time_post, vr = vertical_trajectory(t=time_bounce_ms, r=yc_bounce_mm,a=g)

        yr = 1

        fig3 = plt.figure(3, figsize=(15, 10))

        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,2,3)
        ax3 = plt.subplot(2,2,4)

        ax1.set_xlabel('$t$ $[ms]$')
        ax1.set_ylabel('$R(t)$ $[mm]$')

        ax2.set_xlabel('$t$ $[ms]$')
        ax2.set_ylabel('$x(t)$ $[mm]$')

        ax3.set_xlabel('$t$ $[ms]$')
        ax3.set_ylabel('$y(t)$ $[mm]$')

        ax1.scatter(time_bounce_ms, radius_bounce_mm, marker='.', color='black')
        ax1.plot(time_bounce_ms, np.ones(len(time_bounce_ms))*np.mean(radius_bounce_mm), linestyle='--', color='red')
        ax1.set_title(r'$<R>$ = ' + str(round(np.mean(radius_bounce_mm),3)) + r' $mm$, ' \
                   '$v_{r}$ = ' + str(round(vr,3)) + r' $m/s$, '\
                   '$t_{r}$ = ' + str(round(delta_time_post,3)) + r' $ms$ ')


        ax2.scatter(time_bounce_ms, xc_bounce_mm, marker='.', color='black')
        ax2.plot(time_bounce_ms, x_fit_post, linestyle='--', color='red')
        ax2.set_title(r'$x_{e} - x_{r}$ = ' + str(round(xe - xr,3)) + ' $mm$, ' \
                   '$u$ $=$ ' + str(round(u_post,3)) +' $m/s$ ')


        ax3.scatter(time_bounce_ms, yc_bounce_mm, marker='.', color='black')
        ax3.plot(time_bounce_ms, y_fit_post, linestyle='--', color='red')
        ax3.set_title(r'$y_{e} - y_{r}$ $=$ ' + str(round(ye-yr,3)) + ' $mm$, ' \
                   '$v_{e}$ $=$ ' + str(round(ve,3)) + ' $m/s$, ' \
                   '$g$ = ' + str(round(g,3)) + ' $m/s^2$ ')

        fig3.savefig('bounce.pdf', format='pdf')

    else:
        vr = 0

    txt_file = open("input.txt","w")

    txt_file.write(f"folder = {folder}\n")
    txt_file.write(f"start = {start}\n")
    txt_file.write(f"stop = {stop}\n")
    txt_file.write(f"ceiling = {ceiling}\n")
    txt_file.write(f"interface = {interface}\n")
    txt_file.write(f"wall = {wall}\n")
    txt_file.write(f"threshold = {threshold1} {threshold2}\n")
    txt_file.write(f"px_microns = {px_microns}\n")
    txt_file.close()

    R = np.nanmean(radius_mm[:impact_index])
    hf_ideal = hf_ideal_mum
    hf_real = (wall - interface)*px_microns
    rho_d, mu_d, gamma_d = liquid_properties(liquid = drop)
    rho_f, mu_f, gamma_f = liquid_properties(liquid = film)

    txt_file = open("params.txt","w")

    txt_file.write(f"DROP\n")
    txt_file.write(f"Radius = {R:.3f} mm\n")
    txt_file.write(f"Density = {rho_d:.0f} kg/m3\n")
    txt_file.write(f"Dynamic viscosity = {mu_d*1000:.3f} mPa.s\n")
    txt_file.write(f"Surface tension = {gamma_d*1000:.3f} mN/m\n")
    txt_file.write(f"Impact velocity = {abs(vi):.3f} m/s\n")
    txt_file.write(f"Bounce height = {bounce_height_mm:.3f} mm\n")
    txt_file.write(f"Contact time = {(rebound_frame-impact_frame)*0.1:.3f} ms\n")
    txt_file.write(f"Maximum spread = {max(spread_mm):.3f} mm\n")
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

    # np.savetxt('time_ms.txt', time_ms, fmt='%0.3f')
    # np.savetxt('yc_mm.txt', yc_mm, fmt='%0.3f')

    if bounce_index != None:
        np.savetxt('Tt.txt', [(bounce_image-impact_image)*(1/fps_hz)/np.sqrt((rho_d*((R/1000)**3))/(gamma_d))], fmt='%0.3f')
        np.savetxt('Sr.txt', [max(spread_mm)/R], fmt='%0.3f')
        np.savetxt('e.txt', [vr/(-vi)], fmt='%0.3f')
        np.savetxt('e_bh.txt', [np.sqrt(2*abs(g)*(bounce_height_mm-R)*(10**-3))/(-vi)], fmt='%0.3f')
    else:
        np.savetxt('Tt.txt', [0], fmt='%0.3f')
        np.savetxt('Sr.txt', [0], fmt='%0.3f')
        np.savetxt('e.txt', [0], fmt='%0.3f')

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

def vertical_trajectory1(t=[None], r=[None]):
    linear_params = np.polyfit(t, r, 2)
    rs = (linear_params[0]*(t[0]**2)) + \
         (linear_params[1]*(t[0]**1)) + \
         (linear_params[2]*(t[0]**0))
    a = linear_params[0]*2.0*1000.0
    vs = ((a/1000.0)*t[0]) + linear_params[1]
    h_max_head_mm = linear_params[2] - (((linear_params[1]**2)/(2*a))*1000)

    # print(max(t), a, h_max_head_mm)

    if max(t)<=0:
        tt = t[-1]
        v_star = -np.power(2*a*((1-h_max_head_mm)/1000),1/2)
    else:
        tt = -(linear_params[0]*1000.0)/a
        v_star = +np.power(2*a*((1-h_max_head_mm)/1000),1/2)

    re = ((a/(2.0*1000.0))*(tt**2)) + \
         (linear_params[1]*(tt**1)) + \
         (linear_params[2]*(tt**0))
    ve = ((a/1000.0)*tt) + linear_params[1]
    rfit = ((a/(2.0*1000.0))*(t**2)) + \
           (linear_params[1]*(t**1)) + \
           (linear_params[2]*(t**0))
    d = tt - t[0]

    return rs, vs, re, ve, rfit, d, v_star, a

################################################################################

def vertical_trajectory(t=[None], r=[None], a = None):
    r_mod = r - ((a/(2.0*1000.0))*(t**2))
    linear_params = np.polyfit(t, r_mod, 1)
    rs = ((a/(2.0*1000.0))*(t[0]**2)) + \
         (linear_params[0]*(t[0]**1)) + \
         (linear_params[1]*(t[0]**0))
    vs = ((a/1000.0)*t[0]) + linear_params[0]
    h_max_head_mm = linear_params[1] - (((linear_params[0]**2)/(2*a))*1000)

    # print(max(t), a, h_max_head_mm)

    if max(t)<=0:
        tt = t[-1]
        v_star = -np.power(2*a*((1-h_max_head_mm)/1000),1/2)
    else:
        tt = -(linear_params[0]*1000.0)/a
        v_star = +np.power(2*a*((1-h_max_head_mm)/1000),1/2)

    re = ((a/(2.0*1000.0))*(tt**2)) + \
         (linear_params[0]*(tt**1)) + \
         (linear_params[1]*(tt**0))
    ve = ((a/1000.0)*tt) + linear_params[0]
    rfit = ((a/(2.0*1000.0))*(t**2)) + \
           (linear_params[0]*(t**1)) + \
           (linear_params[1]*(t**0))
    d = tt - t[0]

    return rs, vs, re, ve, rfit, d, v_star

################################################################################

def liquid_properties(liquid = None):

    if liquid == 1:
        rho = 816
        eta = 1*(rho/1000)*(1/1000)
        gamma = 0.017
    elif liquid == 5:
        rho = 960
        eta = 5*(rho/1000)*(1/1000)
        gamma = 0.020
    elif liquid == 10:
        rho = 930
        eta = 10*(rho/1000)*(1/1000)
        gamma = 0.020
    elif liquid == 20:
        rho = 950
        eta = 20*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 35:
        rho = 960
        eta = 35*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 50:
        rho = 960
        eta = 50*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 100:
        rho = 960
        eta = 100*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 200:
        rho = 970
        eta = 200*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 350:
        rho = 970
        eta = 350*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 500:
        rho = 970
        eta = 500*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 1000:
        rho = 971
        eta = 1000*(rho/1000)*(1/1000)
        gamma = 0.021
    elif liquid == 'water':
        rho = 995
        eta = 1*(rho/1000)*(1/1000)
        gamma = 0.072
    else:
        rho = None
        eta = None
        gamma = None

    return rho, eta, gamma

################################################################################

def read_cih():

    data = open(glob.glob('*.cih')[0])
    data_lines = data.readlines()

    rec = int(tuple(data_lines)[15][19:])
    bit = int(tuple(data_lines)[28][21:])
    width = int(tuple(data_lines)[23][14:])
    height = int(tuple(data_lines)[24][15:])

    return rec, bit, width, height

################################################################################
