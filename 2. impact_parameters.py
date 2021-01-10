from FUNC_ import lengthscale_info
from FUNC_ import impact_info

################################################################################

lengthscale_foldername  = r'D:/harddisk_file_contents/color_interferometry/side_view/20201212'
lengthscale_file        = r'px_microns.txt'

px_microns = lengthscale_info(\
                                lengthscale_foldername      = lengthscale_foldername, \
                                lengthscale_file            = lengthscale_file)

################################################################################

folder = r'D:\harddisk_file_contents\color_interferometry\side_view\20201212\oil_1cSt_impact_H_4R_on_100cSt_10mum_run3_'
file   = r'oil_1cSt_impact_H_4R_on_100cSt_10mum_run3_.cih'
drop = '1 cSt oil'
film = '100 cSt oil'
ideal_film_thickness_mum = 10
before_impact = [21, 118]
after_impact = [283, 512]
free_surface = 1005
wall, ceiling = 1000, 680
threshold = 500
radius = 35

impact_info(    folder          = folder, \
                file            = file, \
                drop            = drop, \
                film            = film, \
                hf_ideal_mum    = ideal_film_thickness_mum, \
                before_impact   = before_impact, \
                after_impact    = after_impact, \
                vertical_limits = [wall, ceiling], \
                free_surface    = free_surface, \
                threshold       = threshold, \
                radius          = radius, \
                px_microns      = px_microns)

################################################################################
