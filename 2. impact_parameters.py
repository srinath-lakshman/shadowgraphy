from FUNC_ import lengthscale_info
from FUNC_ import impact_info

import numpy as np

################################################################################

lengthscale_foldername  = r'E:/harddisk_file_contents/color_interferometry/side_view/20201212'
lengthscale_file        = r'px_microns.txt'

px_microns = lengthscale_info(\
                                lengthscale_foldername      = lengthscale_foldername, \
                                lengthscale_file            = lengthscale_file)

################################################################################

folder = r'E:\harddisk_file_contents\color_interferometry\side_view\20201212\oil_1cSt_impact_H_4R_on_100cSt_10mum_run1_'
drop = '1 cSt oil'
film = '100 cSt oil'
wall = 1000

hf_ideal_mum = 10
ceiling = 680
free_surface = 1005

start_frame = 22
end_frame = 500

threshold = [500, 1200]
px_microns = px_microns

impact_info(    folder          = folder, \
                drop            = drop, \
                film            = film, \
                hf_ideal_mum    = hf_ideal_mum, \
                start_frame     = start_frame, \
                end_frame       = end_frame, \
                ceiling         = ceiling, \
                wall            = wall, \
                free_surface    = free_surface, \
                threshold       = threshold, \
                px_microns      = px_microns)

################################################################################
