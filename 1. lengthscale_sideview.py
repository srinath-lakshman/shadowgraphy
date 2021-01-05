from FUNC_ import rectangular_fit

################################################################################

foldername      = r'D:\harddisk_file_contents\color_interferometry\side_view\20201212'
image_filename  = r'reference_lengthscale.tif'
vertical_crop   = [750, 950]

px_microns = rectangular_fit(\
                                foldername      = foldername, \
                                image_filename  = image_filename, \
                                vertical_crop   = vertical_crop)

################################################################################
