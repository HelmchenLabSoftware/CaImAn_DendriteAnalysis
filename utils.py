import numpy as np
from tifffile import imread, imsave
from bokeh.models import HoverTool

def mosaic_to_stack(tiff_file, n_planes, x_crop):
    """
    Convert movie with multiple planes as mosaic (TYX) to ImageJ hyperstack (TZCYXS order)
    tif_file ... tiff file containing movie
    n_planes ... number of planes in Y
    x_crop ... number of pixels (>=1) or fraction (<1) to crop in x
    """
    
    mov = imread(tiff_file)
    
    # check if the number of planes matches the shape of the movie
    if mov.shape[1] % n_planes:
        raise Exception('Number of rows in movie must be divisible by number of planes!')
    rows_per_plane = int(mov.shape[1]/n_planes)
    
    if x_crop < 1:
        x_crop = int(mov.shape[-1]*x_crop)
    
    # stack planes
    plane_list = []
    for ix in range(n_planes):
        plane_list.append(mov[:,ix*rows_per_plane:(ix+1)*rows_per_plane,:x_crop])
        plane_list[ix] = plane_list[ix][None, ...]
        
    stacked = np.vstack(tuple(plane_list)).swapaxes(0, 1)
    # add dimension for channel
    stacked = stacked[:,:,None,:,:]
    
    # write ImageJ HyperStack (TZCYXS order)
    imsave(tiff_file.replace('.tif', '_stacked.tif'), stacked, imagej=True)


def cropTif(fname, crop_pixel):
    """
    Crop TIF file in x and / or y. Save output as *_crop.tif. Only process movies.
    fname ... input TIF file as list
    crop_pixel ... number of pixels in x and y to crop as tuple
    
    returns is_movie (true/false)
    """
    is_movie = True
    # load data
    mov = imread(fname)
    if len(mov.shape) < 3: # not a movie!
        is_movie = False
#         print('%s is not a movie. Skipping.' % (fname))
    else:
        mov = mov[:,crop_pixel[1]:,crop_pixel[0]:]
        imsave(fname.replace('.tif','_crop.tif'), mov)
    return is_movie


def getFramesTif(fname):
    """
    Returns the number of frames in a multi-frame TIF file.
    Return 0 for single-page TIFs.
    """
    # load data
    mov = imread(fname)
    if len(mov.shape) < 3: # not a movie!
        return 0
    else:
        return mov.shape[0]
    
    
def getHover():
    """Define and return hover tool for a Bokeh plot"""
    # Define hover tool
    hover = HoverTool()
    hover.tooltips = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("trial", "@trial_idx (@trial_name)"),
    ]
    return hover