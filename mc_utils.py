import os
import datetime
import json
import numpy as np
from tifffile import imsave
import caiman as cm
from caiman.motion_correction import MotionCorrect

def setupMC(fname, params):
    """
    Configure motion correction oject mc with input filename and parameters.
    
    Return mc
    """
    mov = cm.load(fname)
    mc = MotionCorrect(fname, mov.min(), dview=params['dview'], 
                       max_shifts=params['max_shifts'], 
                       niter_rig=params['niter_rig'], 
                       splits_rig=params['splits_rig'],
                       num_splits_to_process_rig=params['num_splits_to_process_rig'], 
                       strides= params['strides'], 
                       overlaps= params['overlaps'], 
                       splits_els=params['splits_els'],
                       num_splits_to_process_els=params['num_splits_to_process_els'], 
                       upsample_factor_grid=params['upsample_factor_grid'], 
                       max_deviation_rigid=params['max_deviation_rigid'],
                       border_nan=params['border_nan'],
                       shifts_opencv = True, 
                       nonneg_movie = False)
    return mc


def interpolateNans(frame, n=10):
    """
    Interpolate NaN values in frame with average of n closest non-nan pixels
    
    Return interpolated frame
    """
    frame_interp = frame.copy()
    # indices for all NaN pixels
    nan_pixel = np.array(np.where(np.isnan(frame))).T
    if not len(nan_pixel):
        return frame
    # indices for all non-NaN pixels
    valid_pixel = np.array(np.where(~np.isnan(frame))).T
    for pix in nan_pixel:
        # distance between NaN pixel and all valid pixels
        dist = np.linalg.norm(valid_pixel - pix, axis=1)
        # find the closest pixels and get their values in frame
        closest_pixel = valid_pixel[np.argsort(dist)[:n],:]
        closest_pixel_vals = frame[closest_pixel[:,0],closest_pixel[:,1]]
        # replace NaN with average
        frame_interp[pix[0],pix[1]] = np.mean(closest_pixel_vals)

    return frame_interp


def computeMetrics(mc, swap_dim, winsize, resize_fact_flow):
    """
    Compute the quality metrics for the registration.
    """
    
    bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)), np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    
    final_size = np.subtract(mc.total_template_els.shape, bord_px) # remove pixels in the boundaries
    
    tmpl_rig, corr_orig, flows_orig, norms_orig, crispness_orig = \
    cm.motion_correction.compute_metrics_motion_correction(mc.fname[0], final_size[0], final_size[1],
                                                           swap_dim, winsize=winsize, play_flow=False, 
                                                           resize_fact_flow=resize_fact_flow)

    tmpl_rig, corr_rig, flows_rig, norms_rig, crispness_rig = \
    cm.motion_correction.compute_metrics_motion_correction(mc.fname_tot_rig[0], final_size[0], final_size[1],
                                                           swap_dim, winsize=winsize, play_flow=False, 
                                                           resize_fact_flow=resize_fact_flow)

    tmpl_els, corr_els, flows_els, norms_els, crispness_els = \
    cm.motion_correction.compute_metrics_motion_correction(mc.fname_tot_els[0], final_size[0], final_size[1],
                                                           swap_dim, winsize=winsize, play_flow=False, 
                                                           resize_fact_flow=resize_fact_flow)
    
    metrics = {
        'tmpl_rig': tmpl_rig,
        'corr_orig': corr_orig,
        'flows_orig': flows_orig,
        'crispness_orig': crispness_orig,
        'norms_orig': norms_orig,
        
        'corr_rig': corr_rig,
        'flows_rig': flows_rig,
        'crispness_rig': crispness_rig,
        'norms_rig': norms_rig,
        
        'tmpl_els': tmpl_els,
        'corr_els': corr_els,
        'flows_els': flows_els,
        'crispness_els': crispness_els,
        'norms_els': norms_els,
    }
    
    return metrics

def computeMetricsWrapper(mc, swap_dim, winsize, resize_fact_flow):
    """
    Wrapper function for computeMetrics. Used to call computeMetrics for several mc in parallel using starmap.
    
    """
    bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)), np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    metrics = computeMetrics(mc, bord_px, swap_dim, winsize, resize_fact_flow)
    
    return metrics


def removeBoundaryPixels(movie, mc, mc_type):
    """
    Remove the boundary pixels corresponding to the max. shift of the registration.
    """
    # compute borders to exclude
    if mc_type == 'els':
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)), np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        final_size = np.subtract(mc.total_template_els.shape, bord_px)
    elif mc_type == 'rig':
        bord_px = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
        final_size = np.subtract(mc.total_template_rig.shape, bord_px)
    
    # remove pixels in the boundaries
    final_size_x = final_size[0]
    final_size_y = final_size[1]
    max_shft_x = np.int(np.ceil((np.shape(movie)[1] - final_size_x) / 2))
    max_shft_y = np.int(np.ceil((np.shape(movie)[2] - final_size_y) / 2))
    max_shft_x_1 = - ((np.shape(movie)[1] - max_shft_x) - (final_size_x))
    max_shft_y_1 = - ((np.shape(movie)[2] - max_shft_y) - (final_size_y))
    if max_shft_x_1 == 0:
        max_shft_x_1 = None

    if max_shft_y_1 == 0:
        max_shft_y_1 = None
    
    movie = movie[:, max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
    
    if mc_type == 'els':
        mc.total_template_els = mc.total_template_els[max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
    elif mc_type == 'rig':
        mc.total_template_rig = mc.total_template_els[max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
    
    return movie, mc


def runMotionCorrection(fname, params):
    """
    Run motion correction for single input file fname using parameters in params. 
    
    Return motion correction object mc
    """
    
    interp_nans = False
    
    # create mc object
    mc = setupMC(fname, params)
    
    # apply rigid correction
    mc.motion_correct_rigid(save_movie=True)
    
    # apply pw-rigid correction
    mc.motion_correct_pwrigid(save_movie=True, template=mc.total_template_rig, show_template = False)
    
    # load corrected movie - els
    mov_els = cm.load(mc.fname_tot_els[0])
    # remove boundary pixels
    mov_els, mc = removeBoundaryPixels(mov_els, mc, 'els')
    
    # create copy to interpolate NaNs
    mov_els_copy = mov_els.copy()
    if interp_nans:
        for ix in range(mov_els.shape[0]):
            mov_els_copy[ix,:,:] = interpolateNans(mov_els[ix,:,:])
    
    # save pw-rigid corrected movies as TIF
    dummy_fname = 'dummy_%s.tif' % (datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    dummy_fname = os.path.join(params['data_folder'], dummy_fname)
    imsave(dummy_fname, mov_els_copy)
    
    # save pw-rigid corrected and interpolated movie as mmap
    base_name = mc.fname_tot_rig[0][:mc.fname_tot_rig[0].find('_rig_')] + '_els_'
    fname_new = cm.save_memmap([dummy_fname], base_name=base_name, order='F')
    
    # remove previous mmap file and rename TIF file
    os.remove(mc.fname_tot_els[0])
    mc.fname_tot_els = [fname_new]
    os.rename(dummy_fname, fname_new.replace('.mmap', '.tif'))
    
    # save rigid corrected movies as TIF
    imsave(mc.fname_tot_rig[0].replace('.mmap','.tif'), cm.load(mc.fname_tot_rig[0]))
    
    return mc


def writeJsonBadFrames(criterion, thresh, frame_ix, mc, mc_type, data_folder):
    """
    Todo: document me
    
    """
    exclude_info = {"criterion": criterion, 
        "threshold:": thresh, 
        "frames": frame_ix}
    if mc_type == 'els':
        json_fname = mc.fname_tot_els[0].replace('.mmap','') + 'badFrames' + '.json'
    elif mc_type == 'rig':
        json_fname = mc.fname_tot_rig[0].replace('.mmap','') + 'badFrames' + '.json'
    with open(os.path.join(data_folder, json_fname), 'w') as fid:
        json.dump(exclude_info, fid)
    print('Created JSON metadata file %s' % (json_fname))