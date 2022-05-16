# %%
import time
import os
import glob
import fnmatch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import nibabel as nib
import math

if __name__ == "__main__":
    test_dir = '/data1/xhou/ML_CVR/smooth8/ML_data/test'
    # test_dir = '/data1/xhou/ML_CVR/smooth8/ML_data/stroke'
    # test_dir = '/data1/xhou/ML_CVR/smooth8/ML_data/stroke/data'
    # test_dir = '/data1/xhou/ML_CVR/smooth8/ML_data/stroke/notUsed'
    os.chdir(test_dir)

    img_size = (91, 109, 91)
    for subname in glob.glob('3T092*/'):

        img_cvr_ml = np.zeros(img_size)
        img_bat_ml = np.zeros(img_size)

        ##ground truth
        sub_cvr_hc_filepath = os.path.join(test_dir, subname, 'HC', 'CVR_HC_clean.nii')
        img_cvr_hc = nib.load(sub_cvr_hc_filepath).get_fdata().astype(np.float32)

        sub_bat_hc_filepath = os.path.join(test_dir, subname, 'bat_v3', 'bat_HC_clean.nii')
        img_bat_hc = nib.load(sub_bat_hc_filepath).get_fdata().astype(np.float)

        ##relative results
        sub_cvr_rs_filepath = os.path.join(test_dir, subname, 'CVR_mean_temporal_f_clean.nii')
        img_cvr_rs = nib.load(sub_cvr_rs_filepath).get_fdata().astype(np.float)

        sub_bat_rs_filepath = os.path.join(test_dir, subname, 'CVR_voxelshift_etco2_v3_cerebellum', 'BAT_mean_temporal_f_clean.nii')
        img_bat_rs = nib.load(sub_bat_rs_filepath).get_fdata().astype(np.float)

        ##machine-learning results
        sub_cvr_ml_filepath = os.path.join(test_dir, subname, 'ML_result_spatialM_ce_nm_Unet_dual_CVR.nii')
        img_cvr_ml = nib.load(sub_cvr_ml_filepath).get_fdata().astype(np.float)

        sub_bat_ml_filepath = os.path.join(test_dir, subname, 'ML_result_spatialM_ce_nm_Unet_dual_BAT.nii')
        img_bat_ml = nib.load(sub_bat_ml_filepath).get_fdata().astype(np.float)

        psnr_index = psnr(img_cvr_hc[:, :, 44], img_cvr_ml[:, :, 44])

        for index in range(img_cvr_hc.shape[2]):
            b= img_cvr_hc[:, :, index]
            ssim_index = ssim(img_cvr_hc[:, :, index], img_cvr_ml[:, :, index])
            print(ssim_index)
            a= 1

        a = 1


