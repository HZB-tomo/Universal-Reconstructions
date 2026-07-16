# -*- coding: utf-8 -*-
"""
Reconstruction from BAMline at BESSY II

Created on Wednesday 1st June 2022

@author: Ralf F Ziesche
"""

# ----- Library -----
import numpy as np
import astra
import h5py
import matplotlib.pyplot as plt
import tomopy # always import tomopy first and afterwards numpy because of better performance!
from scipy.ndimage import shift
import os
from os.path import isfile, join, exists
from os import listdir, makedirs
import skimage.io as io
import sys
from multiprocessing import Pool
sys.path.insert(0, "T:\\Ralf\\sarepy-master\\")
import psutil
from scipy.ndimage import shift

# ----- Functions -----

# --- load h5 ---
# load full data set
def load_h5_sino_all(path_h5):
    f_before, f_after = 20, 20
    with h5py.File(path_h5, 'r') as f:
        angles = f['entry/instrument/NDAttributes/CT_MICOS_W'][20:-20]
        data = f['entry/data/data']
        sino = data[f_before:-f_after, :, :]
        flat = np.append(data[0:f_before, :, :], data[-f_after:, :, :], 0)
        #energy = f['entry/instrument/NDAttributes/DMM_Energy']
    return angles, flat, sino#, energy


# --- load h5 angles and energy ---
# load full data set
def load_h5_angles_energy(path_h5):
    with h5py.File(path_h5, 'r') as f:
        angles = f['entry/instrument/NDAttributes/CT_MICOS_W'][20:-20]
        energy = f['entry/instrument/NDAttributes/DMM_Energy']
    return angles, energy


# --- load tifs ---
def import_tifs(path):
	onlytif = [f for f in listdir(path+'/') if isfile(join(path+'/',f)) and f[-3:] == 'tif']
	onlytif.sort()
	imy,imz = io.imread(path+'//'+onlytif[0]).shape
	imx = len(onlytif)
	OB = np.zeros((imx,imy,imz), dtype=float)
	i=0
	print(imx,imy,imz)
	for f in onlytif:
	    OB[i,:,:] = io.imread(path+'//'+f)
	    i+= 1
	return OB



# ------------------------------------------------------------------------------

# --- norm best flat old ---
# normalise single sinogram old
def Normalise_sino_best_flat_old(sino, flat):
    sino_norm = np.zeros_like(sino, dtype=np.float32)
    flat = flat.astype(np.float32)
    sino = sino.astype(np.float32)
    best_list = np.zeros(sino.shape[0])
    # flat_best
    for i_0 in range(sino.shape[0]):
        if i_0%10 == 0:
            print("proj: ", i_0)
        best = np.zeros(flat.shape[0])
        proj_best = np.zeros_like(flat, dtype=np.float32)
        for i_1 in range(flat.shape[0]):
            proj_best[i_1] = (sino[i_0]-100)/(flat[i_1]-100)
            best[i_1] = np.std(proj_best[i_1])
        sino_norm[i_0] = proj_best[np.argmin(best)]
        best_list[i_0] = np.argmin(best)
    return sino_norm, best_list

# --- norm best flat ---
# normalise single proj with best flat
def Normalise_proj_best_flat(proj, flat, i=None):
    print("proj: ", i)
    flat = flat.astype(np.float32)
    proj = proj.astype(np.float32)
    # flat_best
    best = np.zeros(flat.shape[0])
    proj_best = np.zeros_like(flat, dtype=np.float32)
    for i_1 in range(flat.shape[0]):
        proj_best[i_1] = (proj-100)/(flat[i_1]-100)
        best[i_1] = np.std(proj_best[i_1])
    proj_norm = proj_best[np.argmin(best)]
    best_list = np.argmin(best)
    if i is not None:
        return i, proj_norm, best_list
    return proj_norm, best_list

# normalise single proj with best flat using a RoI
def Normalise_proj_best_flat_RoI(proj, flat, i=None):
    #flat = flat.astype(np.float32)
    #proj = proj.astype(np.float32)
    # flat_best
    best = np.zeros(flat.shape[0])
    proj_best = np.zeros_like(flat[:,int(flat.shape[1]/4):int(flat.shape[1]-(flat.shape[1]/4)),100:-100], dtype=np.float32)
    for i_1 in range(flat.shape[0]):
        proj_best[i_1] = (proj[int(flat.shape[1]/4):int(flat.shape[1]-(flat.shape[1]/4)),100:-100]-100)/(flat[i_1,int(flat.shape[1]/4):int(flat.shape[1]-(flat.shape[1]/4)),100:-100]-100)
        best[i_1] = np.std(proj_best[i_1])
    # normalise with best flat
    proj_norm = (proj-100)/(flat[np.argmin(best)]-100)
    best_list = np.argmin(best)
    print("proj_norm.dtype: ", proj_norm.dtype)
    if i is not None:
        print("proj", i)
        return i, proj_norm, best_list
    return proj_norm, best_list

# multicore norm with best flat using RoI
def Normalise_proj_best_flat_multi(sino, flat, core_numb=None):
    flat = flat.astype(np.float32)
    sino = sino.astype(np.float32)
    if core_numb is None:
        import psutil
        nProc = psutil.cpu_count()
        print("number of used cores: ",nProc)
    else: nProc = core_numb
    print("used cores: ", nProc)
    pool = Pool(processes=nProc)  # nProc number of used cores
    procs = [pool.apply_async(Normalise_proj_best_flat_RoI, args=(sino[i1,:,:], flat, i1)) \
            for i1 in range(sino.shape[0])]  # for-loop for 2d array
    m = [p.get() for p in procs]  # store multiprocessing results until end of calculation
    pool.close()
    pool.join()
    # sort projections:
    sino_norm = np.zeros_like(sino, np.float32)
    best_flat = np.zeros(sino.shape[0])
    print("sino_norm.dtype: ", sino_norm.dtype)
    for i2 in range(sino.shape[0]):
        sino_norm[int(m[i2][0]),:,:] = np.array(m[i2][1])
        best_flat[int(m[i2][0])] = m[i2][2]
    print("normalise best flat done!")
    return sino_norm, best_flat




# ------------------------------------------------------------------------------
# ----- Ring Filtering ---

# ----- ringe removel -----
from multiprocessing import Pool

# remove ring artefacts
def Ring_removal(sino, sigma_, i):
    print(i)
    print("sigma_: ", sigma_)
    # import used libaries
    import sarepy.prep.stripe_removal_original as srm1
    import sarepy.prep.stripe_removal_improved as srm2
    import sarepy.prep.stripe_removal_former as srm3
    # ring removal
    #sino_ring = srm1.remove_stripe_based_sorting(sino, size=300, dim=1)
    sino_ring = srm1.remove_stripe_based_filtering(sino, sigma=sigma_, size=50, dim=1)
    #sino_ring = srm1.remove_stripe_based_fitting(sino, order=5, sigmax=3, sigmay=3)
    return sino_ring, i

# ----- multiprocessing -----
def Ring_removal_multicore(sino, sigma_, core_numb=None):
    if core_numb is None:
        import psutil
        nProc = psutil.cpu_count()-2
        print("number of used cores: ",nProc)
    else: nProc = core_numb
    pool = Pool(processes=nProc)  # nProc number of used cores
    procs = [pool.apply_async(Ring_removal, args=(sino[:,i1,:], sigma_, i1)) \
            for i1 in range(sino.shape[1])]  # for-loop for 2d array
    #m = np.asarray([p.get() for p in procs])  # store multiprocessing results until end of calculation
    m = [p.get() for p in procs]  # store multiprocessing results until end of calculation
    pool.close()
    pool.join()
    # sort projections:
    result = np.zeros_like(sino)
    for i2 in range(sino.shape[1]):
        result[:,int(m[i2][1]),:] = m[i2][0]
    print("ring filtering done!")
    return result

# ------------------------------------------------------------------------------
# ----- Minlog für absorption coefficient recon -----

def Minlog_sino(sino_norm, pixel_size, core_numb): # pixel_size in cm
    sino_minlog = tomopy.minus_log(sino_norm, ncore=core_numb)/pixel_size
    return sino_minlog.astype(np.float32)


# ------------------------------------------------------------------------------
# ----- Recon single slice -----
def Recon_sino(sino_minlog, angles, center):
    # center_shift because new ASTRA version cant use center
    sino_shift = shift(sino_minlog, [0, (sino_minlog.shape[1] / 2) - center])
    slice = FBP_ASTRA(proj=sino_shift, angle=np.radians(angles), center=center)
    return slice.astype(np.float32)


# ----- find center of rotation (CoR) -----

def entropyMetric(img, i=None, h=1e-5):
    # from Malte
    """Calculate the entropy H for a reconstructed slice using an optimized
    numpy implementation. Code updated for efficiency 2020/09/29.

    metric definition from:
        Donath et al., J. Opt. Soc. Am. A, Vol. 23, No. 5, (2006), p 1048

    :param img: The input 2d image
    :type img: (2d) nd-array
    
    :param i: The index number for the submitted image. Useful for asynchronous
              parallel processing.
    :type i: None or int
    
    :param h: a scaling factor for interval width (default: 8.9e-6)
    :type h: float

    :return: Default: The entropy metric value. If the input i is not None,
             i and the entropy metric value will be returned.
    :rtype: float (or int, float)
    """
    #calculate the edges and bins:
    edge_rise = img.flatten() - h/2
    edge_fall = edge_rise[:] + h
    N = img.size
    bin_bounds = np.concatenate((edge_rise, edge_fall))
    #sort the bin bounds and calculate bin widths
    index_sorted = np.argsort(bin_bounds)
    bin_bounds = bin_bounds[index_sorted]
    delta_g_j = (np.roll(bin_bounds, -1) - bin_bounds)[:-1]
    #calculate the number of points per bin
    arrFlanks = np.ones((2*N), dtype=np.float)
    arrFlanks[N:] = -1
    n_j = np.cumsum(arrFlanks[index_sorted])[:-1]
    #compute and normalize the histogram entropy
    H = -np.sum(np.nan_to_num(delta_g_j * n_j / (h * N) * np.log2(n_j/N)))
    H /= np.log2(N)
    del(arrFlanks, n_j, img, bin_bounds, index_sorted, delta_g_j)
    if i is not None:
        return i, H
    return H


def find_CoR(proj, angles, center_delta, step_size):
    print("find CoR")
    CoR_rough = tomopy.find_center_vo(proj, ind=None, smin=-center_delta, smax=center_delta, srad=6, step=step_size, ratio=0.5, drop=20)
    print("CoR_rough: ", CoR_rough)
    # --- recon different center of rotation ---
    FBP_CoRs = np.zeros([(center_delta*2)*4+1,proj.shape[2],proj.shape[2]])
    delta_CoR = np.zeros([(center_delta*2)*4+1])
    entropy_array = np.zeros_like(delta_CoR)
    # --- use entropy to improve refine CoR ---
    for i in range(FBP_CoRs.shape[0]):
        delta_CoR[i] = CoR_rough - center_delta + step_size*i
        FBP_CoRs[i,:,:] = FBP_ASTRA_GPU(proj[:,int((proj.shape[1]/2)),:],angles,delta_CoR[i], i=None, GPU_numb=None)
        entropy_array[i] = entropyMetric(FBP_CoRs[i,:,:], i=None, h=1e-5)
    CoR_entropy = delta_CoR[np.argmin(entropy_array)]
    print("CoR_entropy: ", CoR_entropy)
    return CoR_entropy





# ----- Filtered Back Projection (FBP) GPU -----

# for single slices
# input: 2D sinogram of the slice; list of angles in rad; center of rotation (CoR)
# output: reconstructed slice


# don't know why needed for CPU
def Recon_sino_CPU(sino_minlog, angles, center):
    # center_shift because new ASTRA version cant use center
    sino_shift = shift(sino_minlog, [0, (sino_minlog.shape[1] / 2) - center])
    slice = FBP_ASTRA_CPU(proj=sino_shift, angle=np.radians(angles))
    return slice.astype(np.float32)

def FBP_ASTRA_CPU(proj,angle,center, i=None):
    #print('Algorithm: FBP ASTRA')
    rec = np.zeros([proj.shape[1],proj.shape[1]], dtype=np.float32)
    #print("rec.shape: ",rec.shape)
    # ----- create geometries and projector
    vol_geom = astra.create_vol_geom(proj.shape[1],proj.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[1], angle)
    proj_id = astra.create_projector('linear',proj_geom, vol_geom)   # not sure why not used
    sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(proj))
    # reconstruction
    recon_id = astra.data2d.create('-vol', vol_geom)
    # ----- create configuration 
    cfg = astra.astra_dict('FBP')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    cfg['option']={}
    # ----- Ram-Lak filter
    cfg['option']['FilterType'] = 'ram-lak'
    # ----- Create and run the algorithm object from the configuration structure
    fbp_id = astra.algorithm.create(cfg)
    astra.algorithm.run(fbp_id)
    # ----- do reconstruction
    rec = astra.data2d.get(recon_id)
    # ----- garbage disposal -----
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.algorithm.delete(fbp_id)
    if i is not None:
        return i, rec
    return rec


# ----- Filtered Back Projection (FBP) GPU -----

# for single slices
# input: 2D sinogram of the slice; list of angles in rad; center of rotation (CoR)
# output: reconstructed slice

'''
# not needed anymore
def Recon_sino_GPU(sino_minlog, angles, center):
    # center_shift because new ASTRA version cant use center
    sino_shift = shift(sino_minlog, [0, (sino_minlog.shape[1] / 2) - center])
    slice = FBP_ASTRA_GPU(proj=sino_shift, angle=np.radians(angles))
    return slice.astype(np.float32)
'''

def FBP_ASTRA_GPU(proj, angle, center, i=None, GPU_numb=None):
    #print('Algorithm: FBP ASTRA')
    print("slice: ", i)
    rec = np.zeros([proj.shape[1],proj.shape[1]], dtype=np.float32)
    cor_shift = (proj.shape[1] / 2 - center)
    #print("cor_shift: ", cor_shift)
    #print("rec.shape: ",rec.shape)
    # ----- create geometries
    vol_geom = astra.create_vol_geom(proj.shape[1],proj.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[1], angle)
    # ----- projection geometry with shifted CoR
    proj_geom_cor = astra.geom_postalignment(proj_geom, cor_shift)
    proj_id_cor = astra.create_projector('cuda',proj_geom_cor, vol_geom)
    sinogram_id = astra.data2d.link('-sino', proj_geom_cor, np.ascontiguousarray(proj))
    #sinogram_id, sinogram = astra.create_sino(np.ascontiguousarray(proj), proj_id_cor)    # not sure why not works!
    # change the proj_geom metadata ...
    astra.data2d.change_geometry(sinogram_id, proj_geom_cor)
    # reconstruction
    recon_id = astra.data2d.create('-vol', vol_geom)
    # ----- create configuration 
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option']={}
    if GPU_numb is not None:
        GPU_use = i % GPU_numb
        #print("GPU_use: ", GPU_use)
        cfg['option']['GPUindex'] = GPU_use
    else: cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
    cfg['option']['MinConstraint'] = 0
    # ----- Create and run the algorithm object from the configuration structure
    fbp_id = astra.algorithm.create(cfg)
    astra.algorithm.run(fbp_id)
    # ----- get reconstruction result
    rec = astra.data2d.get(recon_id)
    # ----- garbage disposal -----
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.projector.delete(proj_id_cor)
    astra.algorithm.delete(fbp_id)
    if i is not None:
        return i, rec
    return rec



def Recon_CoR_multicore_GPU(sino, angles, centers, core_numb=None):
    if core_numb is None:
        import psutil
        nProc = psutil.cpu_count()
        print("number of used cores: ",nProc)
    else: nProc = core_numb
    pool = Pool(processes=nProc)  # nProc number of used cores
    procs = [pool.apply_async(FBP_ASTRA_GPU, args=(sino, angles, centers[i1], i1)) \
            for i1 in range(centers.shape[0])]  # for-loop for 2d array
    #m = np.asarray([p.get() for p in procs])  # store multiprocessing results until end of calculation
    m = [p.get() for p in procs]  # store multiprocessing results until end of calculation
    pool.close()
    pool.join()
    # sort projections:
    result = np.zeros([centers.shape[0],sino.shape[1],sino.shape[1]])
    for i2 in range(centers.shape[0]):
        result[int(m[i2][0]),:,:] = np.array(m[i2][1])
    print("CoR reconstruction done!")
    return result


def Recon_CT_multicore_GPU(sino, angles, center, core_numb=None):
    if core_numb is None:
        import psutil
        nProc = psutil.cpu_count()
        print("number of used cores: ",nProc)
    else: nProc = core_numb
    pool = Pool(processes=nProc)  # nProc number of used cores
    procs = [pool.apply_async(FBP_ASTRA_GPU, args=(sino[:,i1,:], angles, center, i1)) \
            for i1 in range(sino.shape[1])]  # for-loop for 2d array
    #m = np.asarray([p.get() for p in procs])  # store multiprocessing results until end of calculation
    m = [p.get() for p in procs]  # store multiprocessing results until end of calculation
    pool.close()
    pool.join()
    # sort projections:
    result = np.zeros([sino.shape[1],sino.shape[2],sino.shape[2]])
    for i2 in range(sino.shape[1]):
        result[int(m[i2][0]),:,:] = np.array(m[i2][1])
    print("CoR reconstruction done!")
    return result




