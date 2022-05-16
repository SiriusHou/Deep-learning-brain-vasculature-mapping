# %%
import time
import os
import glob
import fnmatch
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import nibabel as nib
import math
from model_spatialS_temporalM_unet_avg3_ssim_sort import ChannelAttnUNet

if __name__ == "__main__":
    test_dir = 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\test'
    os.chdir(test_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\parameter_spatialS_temporalM_unet_avg3_ssim_sort1.pt'
    model = ChannelAttnUNet(in_channels=50, n_classes=1, depth=5, padding=True, up_mode='upconv').to(device)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for subname in glob.glob('*\\'):

            h_result = np.zeros((91, 50))
            test_att_result = np.zeros((96, 112, 91))
            test_result = np.zeros((96, 112, 91))

            tensor_x = []
            tensor_mx = []
            tensor_y = []

            sub_filename = os.path.join(test_dir, subname, 'bold_cere_sort.nii')
            print(sub_filename)
            img = nib.load(sub_filename)
            img_data = img.get_fdata()
            img.uncache()

            for i in range(80):
                train_image = np.squeeze(img_data[:, :, i, :])
                train_image = np.pad(train_image, ((2, 3), (1, 2), (0, 0)),
                                     'constant')  # pad zero surrounding the image
                train_image = np.transpose(train_image, (2, 0, 1))
                tensor_x.append(torch.Tensor(train_image))

            sub_m_filename = os.path.join(test_dir, subname, 'CVR_mean_temporal_f_rp.nii')
            print(sub_m_filename)
            mimg = nib.load(sub_m_filename)
            mimg_data = mimg.get_fdata()
            mimg.uncache()

            for i in range(80):
                train_mimage = np.squeeze(mimg_data[:, :, i])
                train_mimage = np.pad(train_mimage, ((2, 3), (1, 2)),
                                     'constant')  # pad zero surrounding the image
                train_mimage = np.expand_dims(train_mimage, axis=2)
                train_mimage = np.transpose(train_mimage, (2, 0, 1))
                tensor_mx.append(torch.Tensor(train_mimage))

            target_filename = os.path.join(test_dir, subname, 'HC/CVR_HC_clean.nii')
            print(target_filename)
            taimg = nib.load(target_filename)
            taimg_data = taimg.get_fdata()
            taimg.uncache()

            for i in range(80):
                label_image = np.squeeze(taimg_data[:, :, i])
                label_image = np.pad(label_image, ((2, 3), (1, 2)), 'constant')  # pad zero surrounding the image
                tensor_y.append(torch.Tensor(label_image))

            tensor_x_stacked = torch.stack(tensor_x)
            tensor_mx_stacked = torch.stack(tensor_mx)
            tensor_y_stacked = torch.stack(tensor_y)

            test_data = torch.utils.data.TensorDataset(tensor_x_stacked, tensor_mx_stacked, tensor_y_stacked)
            testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)

            i = 0
            for X, X_m, y in testloader:
                X = X.to(device)
                X_m = X_m.to(device)
                #         plt.imshow(X[0][1])
                #         plt.show()
                h, x_h, prediction = model(X, X_m)
                h_result[i, :] = h.cpu().clone().numpy()
                # test_att_result[:, :, i] = np.squeeze(x_h.cpu().clone().numpy())
                test_result[:, :, i] = np.squeeze(prediction.cpu().clone().numpy())
                i += 1

            np.savetxt(os.path.join(test_dir, subname,'attention_spatialS_temporalM_unet_avg3_ssim_sort1.txt'), h_result, fmt='%f')

            # img_att_output = nib.Nifti1Image(test_att_result, taimg.affine)
            # img_att_output.get_data_dtype() == np.dtype(np.int16)
            # nib.save(img_att_output, os.path.join(test_dir, subname, 'ML_attResult_spatialS_temporalM_unet_avg3_ssim_sort1.nii'))

            img_output = nib.Nifti1Image(test_result, taimg.affine)
            img_output.get_data_dtype() == np.dtype(np.int16)
            nib.save(img_output, os.path.join(test_dir, subname, 'ML_result_spatialS_temporalM_unet_avg3_ssim_sort1.nii'))
