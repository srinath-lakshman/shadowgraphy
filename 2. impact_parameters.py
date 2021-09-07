from FUNC_ import info
from FUNC_ import impact_info
# from FUNC_ import inviscid_pool_info

################################################################################

# dir = r'E:\harddisk_file_contents\color_interferometry\side_view'
# dir = r'D:\others\side_view'
dir = r'F:\substrate_independent\r3'

################################################################################

px_microns, global_background = info(dir = dir)

################################################################################

folder  = r'\drop_35cSt'
project = r'\oil_35cSt_impact_H_11R_on_100cSt_10mum_run2_'

# folder  = r'\bond_number_variations\r3'
# project = r'\oil_20cSt_impact_H_11R_on_glass_slide_'

# folder  = r'\H_7R\100cSt_film'
# project = r'\oil_5cSt_impact_H_7R_on_100cSt_600mum_run1_'

# folder = r'\inviscid_pool\Weber10'
# project = r'\oil_500cSt_impact_H_15R_on_1cSt_12mm_run2_'

# folder  = r'\weber_numer_4_match'
# project = r'\oil_5cSt_impact_H_7R_on_10cSt_300mum_run1_'

impact_info(\
             folder     = dir + folder + project,
             px_microns = px_microns,
             global_background = global_background)

# inviscid_pool_info(\
#              folder     = dir + folder + project,
#              px_microns = px_microns,
#              global_background = global_background)

################################################################################
