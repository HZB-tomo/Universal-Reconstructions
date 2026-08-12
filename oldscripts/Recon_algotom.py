#!/usr/bin/env python3

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tifffile
from algotom.io import load_hdf
from algotom.prep import remove_outlier_diffusion, normalization
from algotom.prep.removal import remove_all_stripe, remove_zinger
from algotom.prep.correction import flat_field_correction
from algotom.rec import dfi_reconstruction, fbp_reconstruction
from algotom.util import rotation_axis
import time

def main():
    """
    Complete pipeline for synchrotron tomography reconstruction:
    - Loading HDF5 data
    - Outlier removal
    - Normalization
    - Ring artifact removal
    - Filtered back-projection reconstruction
    - Gray level clipping for 16-bit output
    """
    start_time = time.time()
    
    # ======== Input parameters ========
    # Path to HDF5 file containing synchrotron data
    input_file = "/path/to/your/synchrotron_data.h5"
    
    # Output directory for results
    output_dir = "/path/to/output_directory"
    os.makedirs(output_dir, exist_ok=True)
    
    # HDF5 dataset keys (these may vary based on your synchrotron facility)
    projection_key = "/exchange/data"  # Raw projections
    flat_key = "/exchange/data_white"  # Flat-field images
    dark_key = "/exchange/data_dark"   # Dark-field images
    
    # Reconstruction parameters
    center_of_rotation = None  # Auto-detect if None
    use_gpu = False            # Set to True if GPU available
    filter_name = "hann"       # Options: ram-lak, shepp, hann, hamming, etc.
    
    # Post-processing parameters
    bit_depth = 32              # Output bit depth (32-bit float)
    percentile_low = 0.1        # Low percentile for contrast stretching
    percentile_high = 99.9      # High percentile for contrast stretching
    
    # ======== Load Data ========
    print("Loading data from HDF5 file...")
    
    # Open the HDF5 file
    with h5py.File(input_file, 'r') as hdf:
        # Load projections, flat and dark fields
        projections = np.array(hdf[projection_key])
        flat_fields = np.array(hdf[flat_key])
        dark_fields = np.array(hdf[dark_key])
    
    print(f"Loaded projections with shape: {projections.shape}")
    print(f"Loaded flat fields with shape: {flat_fields.shape}")
    print(f"Loaded dark fields with shape: {dark_fields.shape}")
    
    # ======== Pre-processing ========
    print("Pre-processing data...")
    
    # Average flat and dark fields
    flat_avg = np.mean(flat_fields, axis=0)
    dark_avg = np.mean(dark_fields, axis=0)
    
    # Apply flat-field correction
    sino_norm = flat_field_correction(projections, flat_avg, dark_avg)
    
    # Remove outliers (zingers) from the normalized sinogram
    sino_clean = remove_zinger(sino_norm, size=3, thresh=0.08)
    
    # Select a sinogram for visualization
    mid_slice = sino_clean.shape[1] // 2
    example_sinogram = sino_clean[:, mid_slice, :]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(sino_norm[:, mid_slice, :], cmap='gray')
    plt.title("Original Sinogram")
    plt.subplot(122)
    plt.imshow(example_sinogram, cmap='gray')
    plt.title("After Outlier Removal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_sinogram_cleaning.png"))
    
    # ======== Ring Artifact Removal ========
    print("Applying ring artifact removal...")
    
    # Apply combined ring artifact removal (works for different types of rings)
    sino_ringfree = remove_all_stripe(example_sinogram, 
                                      snr=3.0,            # Signal-to-noise ratio
                                      la_size=61,         # Window size for large stripes
                                      sm_size=21,         # Window size for small stripes
                                      drop_ratio=0.1)     # For removing wide-band stripes
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(example_sinogram, cmap='gray')
    plt.title("Before Ring Removal")
    plt.subplot(122)
    plt.imshow(sino_ringfree, cmap='gray')
    plt.title("After Ring Removal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_ring_removal.png"))
    
    # ======== Center of Rotation Detection ========
    if center_of_rotation is None:
        print("Detecting center of rotation...")
        # Use a method to automatically find the center of rotation
        center_of_rotation = rotation_axis.find_center_vo(example_sinogram)
        print(f"Detected center of rotation: {center_of_rotation}")
    
    # ======== Reconstruction ========
    print("Performing reconstruction...")
    
    # Reconstruct the example sinogram using filtered back-projection
    reconstruction = fbp_reconstruction(
        sino_ringfree,
        center_of_rotation,
        filter_name=filter_name,
        use_gpu=use_gpu
    )
    
    # ======== Post-processing ========
    print("Post-processing reconstructed image...")
    
    # Calculate the percentiles for contrast stretching
    p_low = np.percentile(reconstruction, percentile_low)
    p_high = np.percentile(reconstruction, percentile_high)
    
    # Normalize to the [0, 1] range while preserving 32-bit precision
    reconstruction_normalized = np.clip((reconstruction - p_low) / (p_high - p_low), 0, 1)
    
    # Keep as 32-bit float
    reconstruction_32bit = reconstruction_normalized.astype(np.float32)
    
    # Save final reconstruction
    plt.figure(figsize=(8, 8))
    plt.imshow(reconstruction_32bit, cmap='gray')
    plt.title("Final Reconstruction")
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_final_reconstruction.png"))
    
    # Save reconstructed slice as 32-bit TIFF
    # PIL doesn't support 32-bit float TIFFs well, so use tifffile instead
    import tifffile
    tifffile.imwrite(os.path.join(output_dir, "reconstructed_slice.tif"), reconstruction_32bit))
    
    # ======== Full Volume Reconstruction ========
    print("To reconstruct the full volume, you would need to:")
    print("1. Process all sinograms (all slices)")
    print("2. Apply the same pre-processing and reconstruction to each")
    print("3. Stack the resulting slices to form a 3D volume")
    
    # Uncomment the following to process the full volume (warning: computationally intensive)
    """
    # Process all sinograms
    volume = np.zeros((sino_norm.shape[1], sino_norm.shape[0], sino_norm.shape[0]), dtype=np.float32)
    
    for i in range(sino_norm.shape[1]):
        # Extract sinogram
        sinogram = sino_norm[:, i, :]
        
        # Remove outliers
        sinogram = remove_zinger(sinogram, size=3, thresh=0.08)
        
        # Remove ring artifacts
        sinogram = remove_all_stripe(sinogram, snr=3.0, la_size=61, sm_size=21, drop_ratio=0.1)
        
        # Reconstruct
        recon_slice = fbp_reconstruction(sinogram, center_of_rotation, filter_name=filter_name, use_gpu=use_gpu)
        
        # Store in volume
        volume[i] = recon_slice
        
        # Display progress
        if i % 10 == 0:
            print(f"Processed slice {i}/{sino_norm.shape[1]}")
    
    # Process the full volume for 32-bit output
    p_low = np.percentile(volume, percentile_low)
    p_high = np.percentile(volume, percentile_high)
    volume_32bit = np.clip((volume - p_low) / (p_high - p_low), 0, 1).astype(np.float32)
    
    # Save as a series of 32-bit TIFF files
    for i in range(volume_32bit.shape[0]):
        tifffile.imwrite(os.path.join(output_dir, f"slice_{i:04d}.tif"), volume_32bit[i])
        
    # Optionally save the entire volume as a single 3D TIFF stack
    tifffile.imwrite(os.path.join(output_dir, "volume.tif"), volume_32bit)
    """
    
    elapsed_time = time.time() - start_time
    print(f"Script completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()