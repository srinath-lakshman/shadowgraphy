from FUNC_ import info
from FUNC_ import impact_info

################################################################################

dir = r'F:\harddisk_file_contents\color_interferometry\side_view'

################################################################################

px_microns, global_background = info(dir = dir)

################################################################################

folder  = r'\H_11R\other_10cSt_film'
project = r'\oil_5cSt_impact_H_11R_on_10cSt_300mum_run3_'

impact_info(\
             folder     = dir + folder + project,
             px_microns = px_microns,
             global_background = global_background)

################################################################################
