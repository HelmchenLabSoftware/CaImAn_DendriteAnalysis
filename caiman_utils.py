import os, json
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import estimate_components_quality_auto as estimate_auto


# def evaluateComponents(images, A, C, b, f, YrA, frame_rate, decay_time, gSig, dims, dview, min_SNR, rval_thr,
#                         use_cnn, cnn_thr):
#     """
#     Run component evaluation and return indices of good and bad components.

#     :return:
#     """

#     # Component evaluation
#     idx_comps, idx_comps_bad, SNR_comp, r_values, cnn_preds = estimate_auto(images, A, C, b, f, YrA, frame_rate,
#                                                                             decay_time, gSig, dims, dview=dview,
#                                                                             min_SNR=min_SNR, r_values_min=rval_thr,
#                                                                             use_cnn=use_cnn, thresh_cnn_lowest=cnn_thr)

#     return idx_comps, idx_comps_bad, SNR_comp, r_values, cnn_preds


def runCNMFiterative(images, frame_rate, decay_time, dims, n_processes, K, gSig, merge_thresh, p, dview, rf,
                       stride_cnmf, init_method, alpha_snmf, gnb, method_deconvolution, min_SNR, rval_thr,
                       use_cnn, cnn_thr):
    """
    Iterative CNMF source extraction. First extract spatial and temporal components on patches and combine them.
    For this step deconvolution is turned off (p=0). Then, the components are evaluated and split into good and bad
    components. In the second step, CNMF is re-run on the accepted patches and components are re-evaluated with
    stricter criteria.

    For details of input parameters see the CaImAn documentation.

    :return:
    """

    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    cnmf_0 = cnmf.CNMF(n_processes=n_processes, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p = 0,  dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf,
                    only_init_patch = False, gnb = gnb)
    cnmf_0 = cnmf_0.fit(images)

    # Component evaluation
    idx_comps, idx_comps_bad, SNR_comp, r_values, cnn_preds = \
        evaluate_components(images, cnmf_0.A, cnmf_0.C, cnmf_0.b, cnmf_0.f, cnmf_0.YrA, frame_rate, decay_time, gSig,
                            dims, dview, min_SNR, rval_thr, use_cnn, cnn_thr)

    # Re-run seeded CNMF on accepted patches to refine and perform deconvolution
    A_in, C_in, b_in, f_in = cnmf_0.A[:,idx_comps], cnmf_0.C[idx_comps], cnmf_0.b, cnmf_0.f
    cnmf_1 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                    merge_thresh=merge_thresh,  Ain=A_in, Cin=C_in, b_in = b_in,
                    f_in=f_in, rf = None, stride = None, gnb = gnb,
                    method_deconvolution=method_deconvolution, check_nan = False)

    cnmf_1 = cnmf_1.fit(images)

    # Todo: another round of component evaluation

    return cnmf_1, idx_comps, idx_comps_bad


def runCNMFsingle(images, frame_rate, decay_time, dims, n_processes, K, gSig, merge_thresh, p, dview, rf,
                    stride_cnmf, init_method, alpha_snmf, gnb, method_deconvolution, min_SNR, rval_thr, use_cnn,
                    cnn_thr):
    """


    :return:
    """

    cnmf_0 = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=None,
                            gnb=gnb, rf=rf, memory_fact=1, method_init=init_method, alpha_snmf=alpha_snmf,
                            method_deconvolution=method_deconvolution)
    cnmf_0 = cnmf_0.fit(images)

    # Component evaluation
    idx_comps, idx_comps_bad, SNR_comp, r_values, cnn_preds = \
        evaluate_components(images, cnmf_0.A, cnmf_0.C, cnmf_0.b, cnmf_0.f, cnmf_0.YrA, frame_rate, decay_time, gSig,
                            dims, dview, min_SNR, rval_thr, use_cnn, cnn_thr)

    return cnmf_0, idx_comps, idx_comps_bad


def loadData(fname):
    """
    
    """
    
    Yr, dims, T = cm.load_memmap(fname)
    d1, d2 = dims

    # offset if movie is negative
    if np.min(Yr) < 0:
        Yr = Yr - np.min(Yr)

    # if file is in F order, convert to C order (required for cnmf)
    # TODO: check if this can be done more efficienctly by saving directly in C order
    if np.isfortran(Yr):
        Yr = np.ascontiguousarray(Yr)

#     images = np.reshape(Yr.T, [T] + list(dims), order='F')
#     Y = np.reshape(Yr, dims + (T,), order='F')
    
    return Yr, dims


def getBadFramesByTrial(bad_frames, trial_index):
    """
    Todo: document
    """

    bad_frames_by_trial = dict()
    for ix, i_bad in enumerate(bad_frames):
        trial_index_bad = trial_index[bad_frames[ix]]
        ix_from_trial_start = bad_frames[ix] - np.where(trial_index==trial_index[bad_frames[ix]])[0][0]
        if trial_index_bad in bad_frames_by_trial:
            bad_frames_by_trial[trial_index_bad] = bad_frames_by_trial[trial_index_bad] + [ix_from_trial_start]
        else:
            bad_frames_by_trial[trial_index_bad] = [ix_from_trial_start]
    
    return bad_frames_by_trial


def getBadFrames(fname):
    """
    
    """
    
    bad_frames = json.load(open(fname.replace('.mmap','badFrames.json')))
    return np.array(bad_frames['frames'], dtype='int64')


def removeBadFrames(fname, trial_index, Yr, dims, bad_frames, data_folder):
    """
    
    """

    bad_frames_by_trial = dict()
    
    bad_frames_by_trial = getBadFramesByTrial(bad_frames, trial_index)
    Yr = np.delete(Yr, bad_frames, axis=1)
    trial_index = np.delete(trial_index, bad_frames, axis=0)
    T = Yr.shape[1]
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    print(images.shape)
    # make sure movie is not negative
    add_to_movie = - np.min(images)
        
    base_name = os.path.basename(fname)
    base_name = base_name[:base_name.rfind('__')] + '_remFrames'
    
    fname_new = cm.save_memmap([images], base_name=os.path.join(data_folder, base_name), 
                               add_to_movie=add_to_movie, order='C')
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')
    print('Deleted %1d frames. Saved to new file %s.' % (len(bad_frames), os.path.basename(fname_new)))
    print('Deleted frames:')
    print(bad_frames)

    return images, Y, fname_new, bad_frames_by_trial, trial_index