#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:13:00 2021

@author: Ralf F Ziesche
"""

# ----- Library -----
import astra
import tomopy # always import tomopy first and afterwards numpy because of better performance!
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, makedirs
import skimage.io as io
from skimage.transform import rescale, resize, downscale_local_mean
import numexpr as ne
from astropy.io import fits
from multiprocessing import Pool

# ----- Functions -----

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


def import_tifs_proj(path,number_proj,offset):
	onlytif = [f for f in listdir(path+'/') if isfile(join(path+'/',f)) and f[-3:] == 'tif']
	onlytif.sort()
	imy,imz = io.imread(path+'//'+onlytif[0]).shape
	imx = len(onlytif)
	OB = np.zeros((number_proj,imy,imz), dtype=float)
	i=0
	print(number_proj,imy,imz)
	for f in onlytif:
		if i >= offset and i < (number_proj+offset):
			OB[i-offset,:,:] = io.imread(path+'//'+f)
			#print('i: ',i)
		i+= 1
	return OB


def rearrange_proj(proj,number_proj):
	proj_backward = np.zeros((proj.shape[0],proj.shape[1],proj.shape[2]))
	#print('proj.shape[0]',proj.shape[0])
	for i in range(proj.shape[0]):
		proj_backward[i] = proj[number_proj-1-i]
		#print('i: ',i)
	return proj_backward



def array_binning(arr,binning):
	print('arr.shape: ',arr.shape)
	#check multiple of bin
	if arr.shape[1]%binning != 0:
		for k in range(arr.shape[1]%binning): 
			arr = np.delete(arr,0,1)
	if arr.shape[2]%binning != 0:
		for k in range(arr.shape[2]%binning):
			arr = np.delete(arr,1,2)
	print('arr.shape new: ',arr.shape)
	#binning
	from skimage.transform import rescale, resize, downscale_local_mean
	arr_bin = np.zeros((arr.shape[0],np.int(arr.shape[1]/3),np.int(arr.shape[2]/3)))
	print('arr_bin.shape: ',arr_bin.shape)
	for i in range(arr.shape[0]): 
		arr_bin[i,:,:] = resize(arr[i,:,:], (arr[i,:,:].shape[0]/binning,arr[i,:,:].shape[1]/binning), anti_aliasing=True)
	print('arr_bin.shape new: ',arr_bin.shape)
	return arr_bin


def import_tifs_proj_bin(path,number_proj,offset, binning):
	onlytif = [f for f in listdir(path+'/') if isfile(join(path+'/',f)) and f[-3:] == 'tif']
	onlytif.sort()
	# check multiple of bin
	size_check = io.imread(path+'//'+onlytif[0])
	print('size_check.shape raw: ',size_check.shape)
	if size_check.shape[0]%binning != 0:
		for k in range(size_check.shape[0]%binning): 
			size_check = np.delete(size_check,1,0)
		print('1')
	if size_check.shape[1]%binning != 0:
		for k in range(size_check.shape[1]%binning):
			size_check = np.delete(size_check,0,1)
		print('2')
	print('size_check.shape new: ',size_check.shape)
	# creat empty array for projections
	imy,imz = size_check.shape
	array = np.zeros((number_proj,np.int(imy/binning),np.int(imz/binning)), dtype=float)
	i=0
	print('array.shape: ',array.shape)
	for f in onlytif:
		if i >= offset and i < (number_proj+offset):
			array_bin = io.imread(path+'//'+f)
			if array_bin.shape[0]%binning != 0:
				for k in range(array_bin.shape[0]%binning): 
					array_bin = np.delete(array_bin,1,0)
				#print('1')
			if array_bin.shape[1]%binning != 0:
				for k in range(array_bin.shape[1]%binning):
					array_bin = np.delete(array_bin,0,1)
				#print('2')
			# binning single projections
			array_binned = resize(array_bin,(array_bin.shape[0]/binning,array_bin.shape[1]/binning), anti_aliasing=True)
			array[i-offset,:,:] = array_binned
			print('i: ',i)
		i+= 1
	return array


def import_tifs_OB_dark_bin(path, binning):
	onlytif = [f for f in listdir(path+'/') if isfile(join(path+'/',f)) and f[-3:] == 'tif']
	onlytif.sort()
	# check multiple of bin
	size_check = io.imread(path+'//'+onlytif[0])
	print('size_check.shape raw: ',size_check.shape)
	if size_check.shape[0]%binning != 0:
		for k in range(size_check.shape[0]%binning): 
			size_check = np.delete(size_check,1,0)
		print('1')
	if size_check.shape[1]%binning != 0:
		for k in range(size_check.shape[1]%binning):
			size_check = np.delete(size_check,0,1)
		print('2')
	print('size_check.shape new: ',size_check.shape)
	# creat empty array for projections
	imy,imz = size_check.shape
	array = np.zeros((len(onlytif),np.int(imy/binning),np.int(imz/binning)), dtype=float)
	i=0
	print('array.shape: ',array.shape)
	for f in onlytif:
		array_bin = io.imread(path+'//'+f)
		if array_bin.shape[0]%binning != 0:
			for k in range(array_bin.shape[0]%binning): 
				array_bin = np.delete(array_bin,1,0)
			#print('1')
		if array_bin.shape[1]%binning != 0:
			for k in range(array_bin.shape[1]%binning):
				array_bin = np.delete(array_bin,0,1)
			#print('2')
		# binning single projections
		array_binned = resize(array_bin,(array_bin.shape[0]/binning,array_bin.shape[1]/binning), anti_aliasing=True)
		array[i,:,:] = array_binned
		print('i: ',i)
		i+= 1
	return array




# ----- Noise Filter - "Remove Outlier ..." -----
# same like from ImageJ/Fiji

from skimage.filters import rank
from scipy import ndimage as ndi
from skimage.morphology import disk, ball

def remove_outliers(image, radius=2, threshold=50, i=None):
	footprint = create_circular_mask(radius*2+1,radius*2+1,center=[radius,radius], radius=radius)
	median_filtered = ndi.median_filter(image, footprint=footprint)
	outliers = (
		(image > median_filtered + threshold)
		| (image < median_filtered - threshold)
	)
	output = np.where(outliers, median_filtered, image)
	if i is not None:
		#output_i = np.zeros([output.shape[0]+1,output.shape[1]])
		#output_i[0,0] = i
		#output_i[1:,:] = output
		return i, output
	return output

def remove_outliers_stack(stack, radius, threshold):
	print("stack.dtype: ", stack.dtype)
	tmp_stack = np.zeros_like(stack)
	for i in range(stack.shape[0]):
		tmp_stack[i,:,:] = remove_outliers(stack[i,:,:], radius, threshold)
		#print(i)
	print("Denoising done")
	return tmp_stack


# ----- multiprocessing -----
def remove_outliers_stack_multi(stack, radius, threshold, core_numb=None):
	if core_numb is None:
		import psutil
		nProc = psutil.cpu_count()
		print("number of used cores: ",nProc)
	else: nProc = 30
	pool = Pool(processes=nProc)  # nProc number of used cores
	procs = [pool.apply_async(remove_outliers, args=(stack[i1], radius, threshold, i1)) \
			for i1 in range(stack.shape[0])]  # for-loop for 2d array
	#m = np.asarray([p.get() for p in procs])  # store multiprocessing results until end of calculation
	m = [p.get() for p in procs]  # store multiprocessing results until end of calculation
	pool.close()
	pool.join()
	# sort projections:
	result = np.zeros_like(stack)
	for i2 in range(stack.shape[0]):
		#result[int(m[i2,0,0]),:,:] = m[i2,1:,:]
		result[m[i2][0]] = m[i2][1] 
	print("outlier filtering done!")
	return result


# --- rotation of 3D recon ---
from scipy.ndimage import rotate as rot

def rot_rec3D(tomo, rot_horizontal, rot_vertical_1, rot_vertical_2, i=None):
    if i is not None:
        rot_horizontal = rot_horizontal+180*(i%2)
    # remove nan's:
    tomo = tomopy.remove_nan(tomo, val=0.0)
    # I. rotate tomogram horizontal:
    print("rotate horizontal")
    tomo = rot(tomo,-rot_horizontal,axes=(1,2),reshape=False)#,mode='constant')
    # II. rotate tomogram vertical_1:
    print("rotate vertical 1")
    tomo = rot(tomo,-rot_vertical_1,axes=(0,2),reshape=False)#,mode='constant')
    # III. rotate tomogram vertical_2:
    print("rotate vertical 2")
    tomo = rot(tomo,rot_vertical_2,axes=(0,1),reshape=False,mode='constant')
    return tomo


# ----- image preparation for IMAT MCP -----
# That functions are for the ToF image preparation for 3D reconstruction

def import_fits_IMAT(path):
	# loads all single "energy-fits" in one image stack
	onlyfits = [f for f in listdir(path+'/') if isfile(join(path+'/',f)) and f[-4:] == 'fits']
	onlyfits.sort()
	hdul = fits.open(path + onlyfits[0])
	data = np.rot90(hdul[0].data,k=-1)
	imy,imz = data.shape
	hdul.close()
	imx = len(onlyfits)
	stack = np.zeros((imx,imy,imz), dtype=float)
	tof_data = np.zeros((imx,2))
	i=0
	print(imx,imy,imz)
	for f in onlyfits:
		hdul = fits.open(path+'//'+f)
		stack[i,:,:] = np.rot90(hdul[0].data,k=-1)
		tof_data[i,0] = hdul[0].header[8] #ToF
		tof_data[i,1] = hdul[0].header[9] #timebin
		hdul.close()
		i+= 1
	return stack, tof_data

# 1. 
def remove_nan_zeros(stack):
	# convertes all nan values to zero and replaces the zeros by a 3x3 median value for each energy
	# it's in testing phase
	number_nan = np.abs(np.count_nonzero(~np.isnan(stack[:]))-stack.shape[0]*stack.shape[1]*stack.shape[2])
	print('number nan start: ',number_nan)
	#nan to zeros:
	stack = np.nan_to_num(stack)
	number_zeros = np.count_nonzero(stack[:]==0)
	number_nan = np.count_nonzero(~np.isnan(stack[:]))-stack.shape[0]*stack.shape[1]*stack.shape[2]
	print('number of zeros before: ',number_zeros)
	print('number of nan before: ',number_nan)
	for z in range(stack.shape[0]):
		if(z % 100 == 0):
			print('z: ',z)
		zpos_y,zpos_x = np.where(stack[z]==0)[0],np.where(stack[z]==0)[1]
		#print('zpos_y: ',zpos_y)
		#print('zpos_x: ',zpos_x)
		for i in range(zpos_x.shape[0]):
			# y-range:
			if zpos_y[i]==0:
				y_start = 0
			else:
				y_start = zpos_y[i]-1
			if zpos_y[i]==stack.shape[1]:
				y_end = zpos_y[i]
			else:
				y_end = zpos_y[i]+2
			# x-range:
			if zpos_x[i]==0:
				x_start = 0
			else:
				x_start = zpos_x[i]-1
			if zpos_x[i]==stack.shape[2]:
				x_end = zpos_x[i]
			else:
				x_end = zpos_x[i]+2
			tmp = stack[z,y_start:y_end,x_start:x_end]
			med = np.nanmedian(tmp)
			stack[z,zpos_y,zpos_x] = med
	number_zeros = np.count_nonzero(stack[:]==0)
	number_nan = np.count_nonzero(~np.isnan(stack[:]))-stack.shape[0]*stack.shape[1]*stack.shape[2]
	print('number_zeros: ',number_zeros)
	print('number_nan: ',number_nan)
	return stack


# 2. remove succesive same counts -----
def same_counts_correction(stack):
	# seach for same counts along the energy-axis, which are 3 times repeated
	# a replacement is not written because there are no cases detected yet
	# not usefull yet
	for y in range(stack.shape[1]):
		for x in range(stack.shape[2]):
			for z in range(stack.shape[0]-4):
				tmp = stack[z:z+3,y,x]
				result = np.all(tmp == tmp[0])
				if result:
					print(tmp)
				#else:
					#print('no')


# energy binning of IMAT MCP data
def energy_bin(stack,tof_data,bin_number):
	# check bin number is multiple of energy bins
	if stack.shape[0]%bin_number == 0:
		print('ok')
		stack_bin = np.zeros((int(stack.shape[0]/bin_number),stack.shape[1],stack.shape[2]))
		tof_data_bin = np.zeros((int(stack.shape[0]/bin_number),tof_data.shape[1]))
	else:
		rest = stack.shape[0]%bin_number
		print('rest: ',rest)
		stack_bin = np.zeros((int((stack.shape[0]-rest)/bin_number),stack.shape[1],stack.shape[2]))
		tof_data_bin = np.zeros((int((stack.shape[0]-rest)/bin_number),tof_data.shape[1]))
	print("shape stack_bin: ",stack_bin.shape)
	print("shape tof_data_bin: ",tof_data_bin.shape)
	# make energy bin
	for z in range(stack_bin.shape[0]):
		if(z % 100 == 0):
			print('z: ',z)
		stack_bin[z,:,:] = np.median(stack[z*bin_number:z*bin_number+bin_number,:,:], axis=0)
		tof_data_bin[z,0] = np.mean([tof_data[z*bin_number,0], tof_data[z*bin_number+bin_number,0]])
		tof_data_bin[z,1] = np.sum(tof_data[z*bin_number:z*bin_number+bin_number,1])
	return stack_bin, tof_data_bin

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
        FBP_CoRs[i,:,:] = FBP_ASTRA(proj[:,int((proj.shape[1]/2)),:],angles,delta_CoR[i])
        entropy_array[i] = entropyMetric(FBP_CoRs[i,:,:], i=None, h=1e-5)
    CoR_entropy = delta_CoR[np.argmin(entropy_array)]
    print("CoR_entropy: ", CoR_entropy)
    return CoR_entropy

def create_circular_mask(h, w, center=None, radius=None): #center=[x,y], radius in pixel
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    radius = radius+0.5 # to scale with ImageJ
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

# ----- from Malte - not used yet -----

def __fitCOR(self, metrics, x, savePic=True):
        """__fitCOR tries to fit a 2nd order polynomial around the minimum of
        the metric values for finding the COR.

        :param metrics: The metrics values for the different CoRs.
        :type metrics: :py:class:`np.ndarray`

        :param x: The CoR values corresponding to the entries in the
                  metrics argument.
        :type x: :py:class:`np.ndarray`

        :param savePic: Keyword to enable saving the resulting image to file.
                        (default: True)
        :type savePic: bool

        :return: The fitted x-position of the best centering value.
        :rtype: float
        """
        i0 = max(metrics.argmin() - 6, 0)
        i1 = min(metrics.argmin() + 6, metrics.size - 1)

        xx = np.linspace(x[i0], x[i1], num=5 * (i1 - i0), endpoint=True)
        p = np.polyfit(x[i0:i1], metrics[i0:i1], 3)
        pp = np.polyder(p, 1)
        ppp = np.polyder(p, 2)

        pp /= pp[0]
        x0 = -pp[1] / 2 + np.sqrt(pp[1] ** 2 / 4 - pp[2])
        x1 = -pp[1] / 2 - np.sqrt(pp[1] ** 2 / 4 - pp[2])

        if np.polyval(ppp, x0) > 0 and x[0] < x0 < x[-1]:
            xc = x0
        elif np.polyval(ppp, x1) > 0 and x[0] < x1 < x[-1]:
            xc = x1
        else:
            xc = None

        if savePic and xc is not None:
            y = np.polyval(p, xx)
            lims = np.asarray([np.amin(metrics), np.amax(metrics)])
            lims = (lims[0] - 0.05 * (lims[1] - lims[0]), lims[1]
                    + 0.05 * (lims[1] - lims[0]))
            fig1 = pylabfigure(figsize=(10, 8))
            fig1a = fig1.add_axes([0.17, 0.11, 0.8, 0.84])
            fig1a.plot(x, metrics, 'o', color='#FFA500', label='reco')
            fig1a.plot(xx, y, '-', color='#1F45FC', label='fit')
            if xc is not None:
                fig1a.plot((xc, xc), lims, '-', color='#565051')
            fig1a.set_ylim(lims)
            fig1a.set_xlim([x[0], x[-1]])
            fig1a.legend(loc='upper left')
            fig1a.set_xlabel('centre of reconstruction / pixel')
            fig1a.set_ylabel('normalized histogram entropy')
            fig1.savefig(('{}/tomopy-centering/scan{:d}_metrics.png'
                          '').format(self.procDir, self.scan))

        if xc is None:
            _terminate('Could not fit COR in search region.')
        return xc
    # __fitCOR


#------------------------------------------------------------------------------------------------------

# ----- Tilt-correction -----

# makes a list of all CoR over the slices from two determined CoR

def tilt(y1,y2,center1,center2,im):
	m = (center2-center1)/(y2-y1)
	print('m: ',m)
	n = center1-(m*y1)
	print('n: ', n)
	center_tilt = np.arange(0,im.shape[1],1)*m+n
	return center_tilt




#------------------------------------------------------------------------------------------------------

# ----- Filtered Back Projection (FBP) -----

# for single slices
# input: 2D sinogram of the slice; list of angles in rad; center of rotation (CoR)
# output: reconstructed slice


def FBP_ASTRA(proj,angle,center, i=None, GPU_numb=None):
    #print('Algorithm: FBP ASTRA')
    rec = np.zeros([proj.shape[1],proj.shape[1]], dtype=np.float32)
    #print("rec.shape: ",rec.shape)
    # ----- create geometries
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[1], angle)
    proj_geom['option']={}
    proj_geom['option']['ExtraDetectorOffset'] = (center-proj.shape[1]/2)*np.ones(proj.shape[0])
    vol_geom = astra.create_vol_geom(proj.shape[1],proj.shape[1])
    # create forward projection
    #proj_id = astra.create_projector('cuda',proj_geom, vol_geom)   # not sure why not used
    sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(proj))
    # reconstruction
    recon_id = astra.data2d.create('-vol', vol_geom,0)
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
    # ----- do reconstruction
    rec = astra.data2d.get(recon_id)
    # ----- garbage disposal -----
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.algorithm.delete(fbp_id)
    if i is not None:
        return i, rec
    return rec

# for a full tomogram
# input: full 3D array of projections/sinograms; list of angles in rad; center of rotation (CoR)
# output: 3D array of the reconstructed tomograpy

def FBP2D_ASTRA(proj,angle,center):
    print('Algorithm: FBP2D ASTRA')
    rec_3D = np.zeros([proj.shape[1],proj.shape[2],proj.shape[2]], dtype=np.float32)
    #print("rec_3D.shape: ",rec_3D.shape)
    for i in range(proj.shape[1]):
        if i % 100 == 0: print("slice: ", i)
        # ----- create geometries
        proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[2], angle)
        proj_geom['option']={}
        #proj_geom['option']['ExtraDetectorOffset'] = (int(center)-proj.shape[2]/2)*np.ones(proj.shape[0])
        proj_geom['option']['ExtraDetectorOffset'] = (center-proj.shape[2]/2)*np.ones(proj.shape[0])
        vol_geom = astra.create_vol_geom(proj.shape[2],proj.shape[2])
        # create forward projection
        #proj_id = astra.create_projector('cuda',proj_geom, vol_geom)   # not sure why not used
        sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(proj[:,i,:]))
        # reconstruction
        recon_id = astra.data2d.create('-vol', vol_geom,0)
        # ----- create configuration 
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['option']={}
        cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
        #cfg['option']['MinConstraint'] = 0
        # ----- Create and run the algorithm object from the configuration structure
        fbp_id = astra.algorithm.create(cfg)
        astra.algorithm.run(fbp_id)
        # ----- do reconstruction
        rec = astra.data2d.get(recon_id)
        # ----- garbage disposal -----
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)
        astra.algorithm.delete(fbp_id)
        rec_3D[i,:,:] = rec
    return rec_3D


# for a full tomogram using multiple GPUs
# input: full 3D array of projections/sinograms; list of angles in rad; center of rotation (CoR), number of sued GPUs
# output: 3D array of the reconstructed tomograpy


def FBP2D_ASTRA_multi(proj,angles,CoR,GPU_numb=None):
    if GPU_numb is None:
        GPU_numb = 1
    rec_3D = np.zeros([proj.shape[1],proj.shape[2],proj.shape[2]], dtype=np.float32)
    if GPU_numb == 1:
        print("proj.shape", CoR.shape)
        for i in range(rec_3D.shape[0]):
            if i % 100 == 0: print("slice: ", i)
            rec_3D[i] = FBP_ASTRA(proj[:,i,:],angles,CoR[i])
    else:    
        pool = Pool(processes=GPU_numb)  # nProc number of used cores
        procs = [pool.apply_async(FBP_ASTRA, args=(proj[:,i1,:], angles, CoR[i1], i1, GPU_numb)) \
                                  for i1 in range(proj.shape[1])]  # for-loop for 2d array
        m = [p.get() for p in procs]  # store multiprocessing results until end of calculation
        pool.close()
        pool.join()
        # sort projections:
        for i2 in range(rec_3D.shape[0]):
            rec_3D[m[i2][0]] = m[i2][1]
    print("FBP recon done!")
    return rec_3D




# ----- (SIRT) -----
# new

# for single slices
# input: 2D sinogram of the slice; list of angles in rad; center of rotation (CoR); number of iterations
# output: reconstructed slice


def SIRT_ASTRA(proj,angle,center,niter):
    print('Algorithm: SIRT ASTRA')
    rec = np.zeros([proj.shape[1],proj.shape[1]], dtype=np.float32)
    #print("rec.shape: ",rec.shape)
    # ----- create geometries
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[1], angle)
    proj_geom['option']={}
    proj_geom['option']['ExtraDetectorOffset'] = (center-proj.shape[1]/2)*np.ones(proj.shape[0])
    vol_geom = astra.create_vol_geom(proj.shape[1],proj.shape[1])
    # create forward projection
    #proj_id = astra.create_projector('cuda',proj_geom, vol_geom)   # not sure why not used
    sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(proj))
    # reconstruction
    recon_id = astra.data2d.create('-vol', vol_geom,0)
    # ----- create configuration 
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option']={}
    cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
    cfg['option']['MinConstraint'] = 0
    # ----- Create and run the algorithm object from the configuration structure
    sirt_id = astra.algorithm.create(cfg)
    astra.algorithm.run(sirt_id,niter)
    # ----- do reconstruction
    rec = astra.data2d.get(recon_id)
    # ----- garbage disposal -----
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.algorithm.delete(sirt_id)
    return rec

# for a full tomogram
# input: full 3D array of projections/sinograms; list of angles in rad; center of rotation (CoR), number of iterations
# output: 3D array of the reconstructed tomograpy

def SIRT2D_ASTRA(proj,angle,center,niter):
    print('Algorithm: SIRT2D ASTRA')
    rec_3D = np.zeros([proj.shape[1],proj.shape[2],proj.shape[2]], dtype=np.float32)
    #print("rec_3D.shape: ",rec_3D.shape)
    for i in range(proj.shape[1]):
        if i % 100 == 0: print("slice: ", i)
        # ----- create geometries
        proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[2], angle)
        proj_geom['option']={}
        proj_geom['option']['ExtraDetectorOffset'] = (center-proj.shape[2]/2)*np.ones(proj.shape[0])
        vol_geom = astra.create_vol_geom(proj.shape[2],proj.shape[2])
        # create forward projection
        #proj_id = astra.create_projector('cuda',proj_geom, vol_geom)   # not sure why not used
        sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(proj[:,i,:]))
        # reconstruction
        recon_id = astra.data2d.create('-vol', vol_geom,0)
        # ----- create configuration 
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['option']={}
        cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
        #cfg['option']['MinConstraint'] = 0
        # ----- Create and run the algorithm object from the configuration structure
        sirt_id = astra.algorithm.create(cfg)
        astra.algorithm.run(sirt_id,niter)
        # ----- do reconstruction
        rec = astra.data2d.get(recon_id)
        # ----- garbage disposal -----
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)
        astra.algorithm.delete(sirt_id)
        rec_3D[i,:,:] = rec
    return rec_3D


# ----- (SART) -----
# new

# for single slices
# input: 2D sinogram of the slice; list of angles in rad; center of rotation (CoR); number of iterations
# output: reconstructed slice


def SART_ASTRA(proj,angle,center,niter):
    print('Algorithm: SART ASTRA')
    rec = np.zeros([proj.shape[1],proj.shape[1]], dtype=np.float32)
    #print("rec.shape: ",rec.shape)
    # ----- create geometries
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[1], angle)
    proj_geom['option']={}
    proj_geom['option']['ExtraDetectorOffset'] = (center-proj.shape[1]/2)*np.ones(proj.shape[0])
    vol_geom = astra.create_vol_geom(proj.shape[1],proj.shape[1])
    # create forward projection
    #proj_id = astra.create_projector('cuda',proj_geom, vol_geom)   # not sure why not used
    sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(proj))
    # reconstruction
    recon_id = astra.data2d.create('-vol', vol_geom,0)
    # ----- create configuration 
    cfg = astra.astra_dict('SART_CUDA')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option']={}
    cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
    #cfg['option']['MinConstraint'] = 0
    # ----- Create and run the algorithm object from the configuration structure
    sart_id = astra.algorithm.create(cfg)
    astra.algorithm.run(sart_id,niter)
    # ----- do reconstruction
    rec = astra.data2d.get(recon_id)
    # ----- garbage disposal -----
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.algorithm.delete(sart_id)
    return rec


# for a full tomogram
# input: full 3D array of projections/sinograms; list of angles in rad; center of rotation (CoR), number of iterations
# output: 3D array of the reconstructed tomograpy

def SART2D_ASTRA(proj,angle,center,niter):
    print('Algorithm: SART2D ASTRA')
    rec_3D = np.zeros([proj.shape[1],proj.shape[2],proj.shape[2]], dtype=np.float32)
    #print("rec_3D.shape: ",rec_3D.shape)
    for i in range(proj.shape[1]):
        if i % 100 == 0: print("slice: ", i)
        # ----- create geometries
        proj_geom = astra.create_proj_geom('parallel', 1.0, proj.shape[2], angle)
        proj_geom['option']={}
        proj_geom['option']['ExtraDetectorOffset'] = (center-proj.shape[2]/2)*np.ones(proj.shape[0])
        vol_geom = astra.create_vol_geom(proj.shape[2],proj.shape[2])
        # create forward projection
        #proj_id = astra.create_projector('cuda',proj_geom, vol_geom)   # not sure why not used
        sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(proj[:,i,:]))
        # reconstruction
        recon_id = astra.data2d.create('-vol', vol_geom,0)
        # ----- create configuration 
        cfg = astra.astra_dict('SART_CUDA')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['option']={}
        cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
        #cfg['option']['MinConstraint'] = 0
        # ----- Create and run the algorithm object from the configuration structure
        sart_id = astra.algorithm.create(cfg)
        astra.algorithm.run(sart_id,niter)
        # ----- do reconstruction
        rec = astra.data2d.get(recon_id)
        # ----- garbage disposal -----
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)
        astra.algorithm.delete(sart_id)
        rec_3D[i,:,:] = rec
    return rec_3D

# ----- Forward Projection (FP) / Radon Transform

# for single slices
# input: 2D single slice of tomography; list of angles in rad
# output: 2D single sinogram

def FP_ASTRA(tomo,angle):
    # https://www.astra-toolbox.com/docs/algs/FP_CUDA.html
    sino = np.zeros([angle.shape[0],tomo.shape[1]], dtype=np.float32)
    # create geometries
    proj_geom = astra.create_proj_geom('parallel', 1.0, tomo.shape[1], angle)
    vol_geom = astra.create_vol_geom(tomo.shape[1],tomo.shape[1])
    # store volume
    volume_id = astra.data2d.create('-vol', vol_geom,tomo)
    # creat forward projection
    sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(sino))
    cfg = astra.astra_dict('FP_CUDA')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['VolumeDataId'] = volume_id
    cfg['option']={}
    cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
    # ----- Create and run the algorithm object from the configuration structure
    fp_id = astra.algorithm.create(cfg)
    astra.algorithm.run(fp_id)
    # ----- do reconstruction
    sinogram = astra.data2d.get(sinogram_id)
    # ----- garbage disposal -----
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(volume_id)
    astra.algorithm.delete(fp_id)
    return sinogram


# for a full tomogram
# input: full 3D array of tomography; list of angles in rad
# output: 3D array of all sinograms

def FP2D_ASTRA(tomo,angle):
    # https://www.astra-toolbox.com/docs/algs/FP_CUDA.html
    sinogram_3D = np.zeros([angle.shape[0],tomo.shape[0],tomo.shape[2]], dtype=np.float32)
    #print("sinogram_3D.shape: ", sinogram_3D.shape)
    for i in range(tomo.shape[0]):
        sino = np.zeros([angle.shape[0],tomo.shape[1]], dtype=np.float32)
        # create geometries
        proj_geom = astra.create_proj_geom('parallel', 1.0, tomo.shape[1], angle)
        vol_geom = astra.create_vol_geom(tomo.shape[1],tomo.shape[1])
        # store volume
        volume_id = astra.data2d.create('-vol', vol_geom,tomo[i,:,:])
        # creat forward projection
        sinogram_id = astra.data2d.link('-sino', proj_geom, np.ascontiguousarray(sinogram_3D[:,i,:]))
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectionDataId'] = sinogram_id
        cfg['VolumeDataId'] = volume_id
        cfg['option']={}
        cfg['option']['GPUindex'] = 0       #decide for 0 (GPU1) or 1 (GPU2)
        # ----- Create and run the algorithm object from the configuration structure
        fp_id = astra.algorithm.create(cfg)
        astra.algorithm.run(fp_id)
        # ----- do reconstruction
        sinogram = astra.data2d.get(sinogram_id)
        # ----- garbage disposal -----
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(volume_id)
        astra.algorithm.delete(fp_id)
        sinogram_3D[:,i,:] = sinogram
    return sinogram_3D

# Helpful for ASTRA:
# http://www.ccppetmr.ac.uk/sites/www.ccppetmr.ac.uk/files/ASTRA_Wim_van_Aarle_MIC_SW2015.pdf
# more for Python:
# https://www.astra-toolbox.com/files/misc/ICTMS2019/20190722_ICTMS_ASTRA_workshop_2d.pdf

