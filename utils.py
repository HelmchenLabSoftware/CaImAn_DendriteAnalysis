from tifffile import imread, imsave
from bokeh.models import HoverTool

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