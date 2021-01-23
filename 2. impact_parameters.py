from FUNC_ import lengthscale_info
from FUNC_ import impact_info

################################################################################

lengthscale_foldername  = r'E:/harddisk_file_contents/color_interferometry/side_view/20201212'
lengthscale_file        = r'px_microns.txt'

px_microns = lengthscale_info(\
                               lengthscale_foldername = lengthscale_foldername, \
                               lengthscale_file       = lengthscale_file)

################################################################################

folder = r'E:\harddisk_file_contents\color_interferometry\side_view\20201212\oil_35cSt_impact_H_4R_on_100cSt_600mum_run3_'

impact_info(\
             folder     = folder,
             px_microns = px_microns)

################################################################################
