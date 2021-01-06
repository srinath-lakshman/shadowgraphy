from FUNC_ import lengthscale_info
from FUNC_ import impact_info

################################################################################

lengthscale_foldername  = r'D:/harddisk_file_contents/color_interferometry/side_view/20201212'
lengthscale_file        = r'px_microns.txt'

px_microns = lengthscale_info(\
                                lengthscale_foldername      = lengthscale_foldername, \
                                lengthscale_file            = lengthscale_file)

################################################################################

folder = r'D:\harddisk_file_contents\color_interferometry\side_view\20201212\oil_1cSt_impact_H_4R_on_100cSt_10mum_run1_'
file   = r'oil_1cSt_impact_H_4R_on_100cSt_10mum_run1_.cih'
before_impact = [13, 116]
after_impact = [282, 510]
wall, ceiling = 1000, 680

impact_info(    folder          = folder, \
                file            = file, \
                before_impact   = before_impact, \
                after_impact    = after_impact, \
                vertical_limits = [wall, ceiling], \
                px_microns      = px_microns)

################################################################################
