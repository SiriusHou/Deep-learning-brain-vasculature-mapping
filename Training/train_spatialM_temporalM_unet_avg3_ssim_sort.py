import time
import os
import glob
from pytorch_msssim import msssim, ssim
from torch.autograd import Variable
import fnmatch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import nibabel as nib
import math
from model_spatialM_temporalM_unet_avg3_ssim_sort import ChannelAttnUNet

# cwd = os.getcwd()
# print(cwd)

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
                    if fnmatch.fnmatch(subfile, 'bold_spatialM_temporalM_sort_[0]*'):
                        sub_filepath = os.path.join(data_root, subname, subfile)
                        self.input_IDs.append(sub_filepath)

                        # sub_filepath = os.path.join(data_root, subname, subfile)
                        # self.input_IDs.append(sub_filepath)

                        parts = sub_filepath.split('\\')
                        parts_prefix = '\\'.join(parts[:-1])
                        parts_end = parts[-1][-7:]

                        self.label_IDs.append(os.path.join(parts_prefix, 'HC', 'CVR_HC_clean_' + parts_end))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        'initialization'
        input_x = []
        label_y = []

        # Select input
        input_ID = self.input_IDs[index]
        # print(input_ID)

        # Load input
        img = nib.load(input_ID)
        train_image = img.get_fdata()
        train_image = train_image.astype(np.float32)
        img.uncache()

        train_image = np.transpose(train_image, (3, 2, 0, 1))

        A = torch.Tensor(train_image).type(torch.FloatTensor)

        # Select label
        label_ID = self.label_IDs[index]
        # print(label_ID)

        # Load label
        taimg = nib.load(label_ID)
        label_image = taimg.get_fdata()
        label_image = label_image.astype(np.float32)
        taimg.uncache()

        B = torch.Tensor(label_image).type(torch.FloatTensor)

        return [A, B]


# %%

if __name__ == "__main__":

    validation_split = 0.1
    random_seed = 40
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    training_set = CVRDataset('d:\\xhou4\\ML_CVR\\smooth8\\ML_data\\train')
    training_set_size = len(training_set)
    indices = list(range(training_set_size))
    split = int(np.floor(validation_split * training_set_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Generators
    train_loader = data.DataLoader(training_set, batch_size=16, sampler=train_sampler, num_workers=12)
    validation_loader = data.DataLoader(training_set, batch_size=16, sampler=valid_sampler, num_workers=12)

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model = ChannelAttnUNet(spatial_channels=20, temporal_channels=25, n_classes=1, depth=5, padding=True, up_mode='upconv').to(device)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ChannelAttnUNet(spatial_channels=20, temporal_channels=25, n_classes=1, depth=5, padding=True, up_mode='upconv')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # enabling data parallelism

    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss()
    criterion_cc = nn.CrossEntropyLoss()
    epochs = 1
    print_every = 10

    train_loss_list = []
    test_loss_list = []

    for t in range(epochs):

        train_loss = 0.0
        running_loss = 0.0

        step_count = 0.0

        test_loss = 0.0
        test_total_loss = 0.0
        test_step_count = 0.0

        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)  # [N, SC, TC, H, W]
            y = y.to(device)  # [N, H, W] with class indices (0, 1)
            h, prediction = model(X)  # [N, H, W]

            y = y.unsqueeze(1)

            # ssim_index_temporal = []
            # for jj in range(X.shape[1]):
            #     ssim_output = []
            #     for kk in range(X.shape[2]):
            #         X_layer = X[:, jj, kk, :, :].unsqueeze(1)
            #         ssim_index = ssim(y, X_layer, window_size=25, radius=4, size_average=None, val_range=10)
            #         ssim_index = ssim_index.unsqueeze(1).unsqueeze(2)
            #         ssim_output.append(ssim_index)
            #     ssim_index_temporal.append(torch.cat(ssim_output, dim=2))
            # ssim_index_all = torch.cat(ssim_index_temporal, dim=1)
            #
            # loss_ssim = criterion(h, ssim_index_all)

            #L1 or L2 loss
            loss_l1 = criterion(prediction, y)
            loss = loss_l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            if math.isnan(loss.item()) == False:
                train_loss += loss.item()
                step_count += 1

            if (i + 1) % print_every == 0:
                running_loss += train_loss
                print('Epoch number {}, Step [{}/{}], Training Loss: {:.4f}, Training l1 Loss: {:.4f} '
                      .format(t + 1, i + 1, len(train_loader), loss.item(), loss_l1.item()))
                train_loss = 0.0

                # validation
                model.eval()
                with torch.no_grad():
                    for X, y in validation_loader:
                        X = X.to(device)  # [N, C, H, W]
                        y = y.to(device)  # [N, H, W] with class indices (0, 1)
                        h, prediction = model(X)  # [N, H, W]

                        y = y.unsqueeze(1)

                        # ssim_index_temporal = []
                        # for jj in range(X.shape[1]):
                        #     ssim_output = []
                        #     for kk in range(X.shape[2]):
                        #         X_layer = X[:, jj, kk, :, :].unsqueeze(1)
                        #         ssim_index = ssim(y, X_layer, window_size=25, radius=4, size_average=None, val_range=10)
                        #         ssim_index = ssim_index.unsqueeze(1).unsqueeze(2)
                        #         ssim_output.append(ssim_index)
                        #     ssim_index_temporal.append(torch.cat(ssim_output, dim=2))
                        # ssim_index_all = torch.cat(ssim_index_temporal, dim=1)
                        #
                        # loss_ssim = criterion(h, ssim_index_all)

                        # L1 or L2 loss
                        loss_l1 = criterion(prediction, y)
                        loss = loss_l1

                        if math.isnan(loss.item()) == False:
                            test_loss = loss.item()
                            test_step_count += 1

                        test_total_loss += test_loss
                        test_loss = 0

        test_loss_list.append(test_total_loss / test_step_count)
        train_loss_list.append(running_loss / step_count)

    ## Plotting batch-wise train loss curve:
    plt.plot(train_loss_list, '-o', label='train_loss', color='blue')
    plt.plot(test_loss_list, '-o', label='valid_loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('d:\\xhou4\\ML_CVR\\smooth8\\ML_data\\train\\valid_spatialM_temporalM_unet_sort_softAttn.png', dpi=300)
    plt.show()

    model_name = 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\parameter_spatialM_temporalM_unet_sort_softAttn.pt'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.cpu().state_dict(), model_name)
    else:
        torch.save(model.cpu().state_dict(), model_name)

