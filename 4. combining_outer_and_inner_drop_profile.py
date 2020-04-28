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

# F:\color_interferometry\bottom_view\20200114\experiment\lower_speed_mica_run1\info\colors\lower_speed_mica_run1_000078

################################################################################

start_number = int('078')
end_number   = int('107')

n = end_number - start_number + 1
fps = 4000
t_microns = np.arange(0,n,1)*(np.power(10,6)/fps)

plt.figure(1, figsize=(3,6))

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)

for i in range(start_number, end_number+1):
# for i in range(k, k+1):
    print(i)
    os.chdir(r'F:/color_interferometry/side_view/20200114/experiment/lower_speed_mica_run1')
    outer_txtfiles = np.loadtxt('outer_profile_' + f'{i:06d}' + '.txt')
    os.chdir(r'F:/color_interferometry/bottom_view/20200114/experiment/lower_speed_mica_run1/info/colors/lower_speed_mica_run1_' + f'{i:06d}')
    inner_txtfiles = np.loadtxt('profile_unfiltered.txt')

    ax1.plot(outer_txtfiles[:,0], outer_txtfiles[:,1], linestyle='-', color='black')
    ax1.scatter(outer_txtfiles[:,0], outer_txtfiles[:,1], marker='.', color='red')
    ax1.set_aspect(1)
    ax1.set_xlim(-0.25,1.50)
    ax1.set_ylim(-0.25,2.25)
    ax1.set_xlabel('')
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_ylabel(r'h [$mm$]')
    ax1.axhline(y=0,linestyle='--',color='black')
    ax1.axvline(x=0,linestyle='--',color='black')

    ax2.plot(inner_txtfiles[:,0], inner_txtfiles[:,1]/1000, linestyle='-', color='black')
    ax2.scatter(inner_txtfiles[:,0], inner_txtfiles[:,1]/1000, marker='.', color='red')
    ax2.set_aspect((25*1000)/75, 'box')
    ax2.set_xlim(-0.25,1.50)
    ax2.set_ylim(-0.0025,0.005)
    ax2.set_xlabel(r'r [$mm$]')
    ax2.set_ylabel(r'h [$\mu m$]')
    ax2.set_yticks([-0.0025, 0, 0.0025, 0.005])
    ax2.set_yticklabels(['-2.5', '0', '+2.5', '+5'])
    ax2.axhline(y=0,linestyle='--',color='black')
    ax2.axvline(x=0,linestyle='--',color='black')

    save_folder = r'F:/color_interferometry/side_view/20200114/experiment/lower_speed_mica_run1/full_profiles/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    os.chdir(save_folder)
    plt.savefig('full_profile_' + f'{i:06d}' + '.png')
    ax1.cla()
    ax2.cla()
    if not os.path.exists(os.getcwd() + '/outer_profiles'):
        os.makedirs(os.getcwd() + '/outer_profiles')
    os.chdir(os.getcwd() + '/outer_profiles')
    np.savetxt('outer_profiles_microns_' + f'{i:06d}' + '.txt', np.column_stack([outer_txtfiles[:,0]*1000, outer_txtfiles[:,1]*1000]), fmt='%1.6f')
    np.savetxt('time_microns.txt', np.column_stack([t_microns]), fmt='%d')
    os.chdir('..')
    if not os.path.exists(os.getcwd() + '/inner_profiles'):
        os.makedirs(os.getcwd() + '/inner_profiles')
    os.chdir(os.getcwd() + '/inner_profiles')
    np.savetxt('inner_profiles_microns_' + f'{i:06d}' + '.txt', np.column_stack([inner_txtfiles[:,0]*1000, inner_txtfiles[:,1]]), fmt='%1.6f')
    np.savetxt('time_microns.txt', np.column_stack([t_microns]), fmt='%d')
