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
from model_spatialM_temporalM_dualUnet_sort import DualUNet
class CVRDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_root):
        'initialization'
        self.input_IDs = []
        self.minput_IDs = []
        self.label_IDs = []
        'Find the file location'
        for subname in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, subname)):
                for subfile in os.listdir(os.path.join(data_root, subname)):
                    if fnmatch.fnmatch(subfile, 'bold_spatialM_temporalM_sort_[0-9][0-9][0-9].nii'):
                        sub_filepath = os.path.join(data_root, subname, subfile)
                        self.input_IDs.append(sub_filepath)
                        parts = sub_filepath.split('\\')
                        parts_prefix = '\\'.join(parts[:-1])
                        parts_end = parts[-1][-7:]


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select input
        input_ID = self.input_IDs[index]

        parts = input_ID.split('.')
        ID = parts[0]

        # Load input
        img = nib.load(input_ID)
        train_image = img.get_fdata()
        train_image = train_image.astype(np.float32)
        img.uncache()

        train_image = np.transpose(train_image, (3, 2, 0, 1))

        A_affine = img.affine
        A = torch.Tensor(train_image).type(torch.FloatTensor)

        return [ID, A, A_affine]
# %%

if __name__ == "__main__":

    test_dir = 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\test'
    os.chdir(test_dir)

    # Input test data
    test_set = CVRDataset(test_dir)
    test_set_size = len(test_set)

    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_path = 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\parameter_spatialM_temporalM_sort_dualUnet_intLoss.pt'
    model = DualUNet(spatial_channels=10, temporal_channels=30, n_classes=1, padding=True, up_mode='upconv').to(device)
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for X_ID, X, X_affine in test_loader:

            X_dir = X_ID[0]
            X = X.to(device)  # [N, SC, TC, H, W]
            X_spa, prediction = model(X)  # [N, H, W]

            prediction = np.squeeze(prediction.cpu().clone().numpy())
            X_affine = np.squeeze(X_affine.cpu().clone().numpy())
            img_output = nib.Nifti1Image(prediction, X_affine)
            img_output.get_data_dtype() == np.dtype(np.int16)
            nib.save(img_output, '_'.join([X_dir, 'ML_dualUnet_result.nii']))

            X_spa = np.squeeze(X_spa.cpu().clone().numpy())
            X_spa = np.transpose(X_spa, (1, 2, 0))
            img_output = nib.Nifti1Image(X_spa, X_affine)
            img_output.get_data_dtype() == np.dtype(np.int16)
            nib.save(img_output, '_'.join([X_dir, 'ML_dualUnet_attn.nii']))


    img_size = (91, 109, 91)
    for subname in glob.glob('*\\'):

        img_data_3D = np.zeros(img_size)
        for subfile in os.listdir(os.path.join(test_dir, subname)):
            if fnmatch.fnmatch(subfile, 'bold_spatialM_temporalM_sort_*_ML_dualUnet_result.nii'):

                parts = subfile.split('_')
                ID = int(parts[4])-1

                sub_filepath = os.path.join(test_dir, subname, subfile)
                img = nib.load(sub_filepath)
                img_data = img.get_fdata()
                img.uncache()

                img_data_3D[:, :, ID] = img_data[2:-3, 1:-2]

        img_output = nib.Nifti1Image(img_data_3D, img.affine)
        img_output.get_data_dtype() == np.dtype(np.int16)
        nib.save(img_output, os.path.join(test_dir, subname, 'ML_result_spatialM_temporalM_dualUnet_sort.nii'))

