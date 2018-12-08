import numpy as np
from tifffile import imread, imsave

# Bokeh imports
from bokeh.plotting import Figure, show
from bokeh.models import Range1d, CrosshairTool, HoverTool, Legend
from bokeh.models.sources import ColumnDataSource

def mosaicToStack(tiff_file, n_planes, x_crop):
    """
    Convert movie with multiple planes as mosaic (TYX) to ImageJ hyperstack (TZCYXS order)
    tif_file ... tiff file containing movie
    n_planes ... number of planes in Y
    x_crop ... number of pixels (>=1) or fraction (<1) to crop in x
    
    Returns: stacked TIFF file name
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
#     stacked = stacked[:,:,None,:,:]
    
    # write ImageJ HyperStack (TZCYXS order)
    outfile = tiff_file.replace('.tif', '_stacked.tif')
    imsave(outfile, stacked, imagej=True)
    return outfile


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


def plotTimeseries(p, t, y, legend=None, stack=True, xlabel='', ylabel='', output_backend='canvas', 
                   trial_index=None, trial_names_frames=None):
    """
    Plot a timeseries in Figure p using the Bokeh library
    
    Input arguments:
    p ... Bokeh figure
    t ... 1d time axis vector (numpy array)
    y ... 2d data numpy array (number of traces x time)
    legend ... list of items to be used as figure legend
    stack ... whether to stack traces or nor (True / False)
    xlabel ... label for x-axis
    ylabel ... label for y-axis
    output_backend ... 'canvas' or 'svg'
    trial_index ... trial index for each frame
    trial_names_frames ... trial name for each frame
    """
    
    colors_list = ['red', 'green', 'blue', 'yellow', 'cyan', 'orange', 'magenta', 'black', 'gray']
    p.add_tools(CrosshairTool(), getHover())
    
    offset = 0
    for i in range(y.shape[0]):
        if len(colors_list) < i+1:
            colors_list = colors_list + colors_list
        
        plot_trace = y[i, :]
        if stack:
            plot_trace = plot_trace - min(plot_trace) + offset
            offset = max(plot_trace)
        
        # create ColumnDataSource
        data = {
            'x': t, 
            'y': plot_trace,
            'trial_idx': trial_index,
            'trial_name': trial_names_frames
        }
        data_source = ColumnDataSource(data)

        # add line
        p.line('x', 'y', source=data_source, line_width=2, legend=legend[i], color=colors_list[i])
        
#     p.legend.location = (0,-30)
    p.legend.click_policy="hide"
    
    # format plot
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel
    
    p.x_range = Range1d(np.min(t), np.max(t))
    
    p.background_fill_color = None
    p.border_fill_color = None
    
    p.output_backend = output_backend

    show(p)
    
    return p