"""

Some modified plotting utilities from the CaImAn analysis suite. 

"""

from builtins import str
from builtins import range
from past.utils import old_div

import base64
import cv2
from IPython.display import HTML
from math import sqrt, ceil
import matplotlib as mpl
from matplotlib.widgets import Slider
import numpy as np
import pylab as pl
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.filters import median_filter
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from skimage.measure import find_contours
import sys
from tempfile import NamedTemporaryFile
from warnings import warn

import bokeh
import bokeh.plotting as bpl
from bokeh.layouts import gridplot
from bokeh.models import CustomJS, ColumnDataSource, Range1d

from caiman.base.rois import com


def nb_view_patches(Yr, idx, A, C, b, f, d1, d2, YrA=None, image_neurons=None, denoised_color=None, cmap='Greys256', title=''):
    """
    Interactive plotting utility for ipython notebook

    Args:
        Yr: movie
            
        idx: indices of components to plot

        A,C,b,f: outputs of matrix factorization algorithm

        d1,d2: dimensions of movie (x and y)

        YrA:   ROI filtered residual as it is given from update_temporal_components

        image_neurons: image to be overlaid to neurons (for instance the average)

        denoised_color: color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: name of colormap (e.g. 'viridis') used to plot image_neurons
    """
    
    A = A[:,idx]
    C = C[idx]
    YrA = YrA[idx]
    
    # compute component images
    component_images = []
    A_dense = A.todense()
    for ix in range(len(idx)):
        component_images.append(np.flipud(np.array(np.reshape(A_dense[:,ix], (d1,d2), order='F'))))

    nr, T = C.shape
    nA2 = np.ravel(np.power(A, 2).sum(0)) if type(
        A) == np.ndarray else np.ravel(A.power(2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                       (A.T * np.matrix(Yr) -
                        (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                        A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    source_line = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    source_line_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source_img = ColumnDataSource(data=dict(img=[component_images[0]]))
    source_img_ = ColumnDataSource(data=dict(img=component_images))
    
    if Y_r.shape[0] > 1:
    
        callback = CustomJS(args=dict(source_line=source_line, source_line_=source_line_, source_img=source_img, source_img_=source_img_), code="""
            var line_data = source_line.data;
            var line_data_ = source_line_.data;
            var img_data = source_img.data;
            var img_data_ = source_img_.data;
            var f = cb_obj.value - 1;

            var x = line_data['x'];
            var y = line_data['y'];
            var y2 = line_data['y2'];

            for (i = 0; i < x.length; i++) {
                    y[i] = line_data_['z'][i+f*x.length];
                    y2[i] = line_data_['z2'][i+f*x.length];
                }

            img = img_data_['img']

            source_img.data['img'] = [img[f]]

            source_line.change.emit();
            source_img.change.emit();
        """)
    
    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source_line, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source_line, line_width=1,
                  line_alpha=0.6, color=denoised_color)
    
    plot1 = bpl.figure(plot_width=image_neurons.shape[1]*2, plot_height=image_neurons.shape[0]*2, 
                       x_range = [0, image_neurons.shape[1]], y_range = [0, image_neurons.shape[0]], 
                       toolbar_location="below", title=title)
    
    plot1.image('img', source=source_img, x=0, y=0, dw=image_neurons.shape[1], dh=image_neurons.shape[0], palette=cmap)
    
    if Y_r.shape[0] > 1:
        slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1, title="Neuron Number", callback=callback)
        grid = gridplot([[plot1, None], [plot, slider]], sizing_mode='fixed', toolbar_location='left')
    else:
        grid = gridplot([[plot1, None], [plot, None]], sizing_mode='fixed', toolbar_location='left')
    
    bpl.show(grid)
    
    return component_images
