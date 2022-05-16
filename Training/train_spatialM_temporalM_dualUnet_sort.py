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
from model_spatialM_temporalM_dualUnet_sort import DualUNet

# cwd = os.getcwd()
# print(cwd)

class CVRDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_root):
        'initialization'
        self.input_IDs = []
        self.minput_IDs = []
        self.label_IDs = []
        self.mlabel_IDs = []
        'Find the file location'
        for subname in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, subname)):
                for subfile in os.listdir(os.path.join(data_root, subname)):
                    if fnmatch.fnmatch(subfile, 'bold_spatialM_temporalM_[0]*'):
                        sub_filepath = os.path.join(data_root, subname, subfile)
                        self.input_IDs.append(sub_filepath)

                        # sub_filepath = os.path.join(data_root, subname, subfile)
                        # self.input_IDs.append(sub_filepath)

                        parts = sub_filepath.split('\\')
                        parts_prefix = '\\'.join(parts[:-1])
                        parts_end = parts[-1][-7:]

                        # self.minput_IDs.append(os.path.join(parts_prefix, 'bold_spatialM_' + parts_end))
                        self.label_IDs.append(os.path.join(parts_prefix, 'HC', 'CVR_HC_clean_' + parts_end))
                        self.mlabel_IDs.append(os.path.join(parts_prefix, 'HC', 'HC_spatialM_' + parts_end))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

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

        # # Select mean input
        # minput_ID = self.minput_IDs[index]
        #
        # # Load mean input
        # mimg = nib.load(minput_ID)
        # train_mimage = mimg.get_fdata()
        # train_mimage = train_mimage.astype(np.float32)
        # mimg.uncache()
        #
        # train_mimage = np.transpose(train_mimage, (2, 0, 1))
        #
        # A_m = torch.Tensor(train_mimage).type(torch.FloatTensor)

        # Select label
        label_ID = self.label_IDs[index]
        # print(label_ID)

        # Load label
        taimg = nib.load(label_ID)
        label_image = taimg.get_fdata()
        label_image = label_image.astype(np.float32)
        taimg.uncache()

        B = torch.Tensor(label_image).type(torch.FloatTensor)

        # Select intermediate target
        mlabel_ID = self.mlabel_IDs[index]

        # Load intermediate target
        mtaimg = nib.load(mlabel_ID)
        label_mimage = mtaimg.get_fdata()
        label_mimage = label_mimage.astype(np.float32)
        mtaimg.uncache()

        label_mimage = np.transpose(label_mimage, (2, 0, 1))

        B_m = torch.Tensor(label_mimage).type(torch.FloatTensor)

        return [A, B, B_m]


# %%

if __name__ == "__main__":

    validation_split = 0.05
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
    train_loader = data.DataLoader(training_set, batch_size=20, sampler=train_sampler, num_workers=10)
    validation_loader = data.DataLoader(training_set, batch_size=20, sampler=valid_sampler, num_workers=10)

    # #single gpu
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model = DualUNet(spatial_channels=10, temporal_channels=30, n_classes=1, padding=True, up_mode='upconv').to(device)

    #multiple gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DualUNet(spatial_channels=10, temporal_channels=30, n_classes=1, padding=True, up_mode='upconv')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # enabling data parallelism

    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss()
    criterion_cc = nn.CrossEntropyLoss()
    epochs = 25
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

        for i, (X, y, y_m) in enumerate(train_loader):
            X = X.to(device)  # [N, SC, TC, H, W]
            # X_m = X_m.to(device)  # [N, SC, H, W]
            y = y.to(device)  # [N, H, W]
            y_m = y_m.to(device)  # [N, SC, H, W]

            X_spa, prediction = model(X)  # [N, H, W]

            y = y.unsqueeze(1)

            #intermediate loss
            loss_l1_inter = criterion(X_spa, y_m)
            # l1_inter = 0
            # for kk in range(X_spa.shape[1]):
            #     X_layer = X_spa[:, kk, :, :].unsqueeze(1)
            #     l1_inter = l1_inter+criterion(X_layer, y)
            #
            # loss_l1_inter = l1_inter/X_spa.shape[1]

            #L1 or L2 loss
            loss_l1 = criterion(prediction, y)
            # loss = loss_l1 + 100 * loss_l1_inter
            loss = loss_l1 + loss_l1_inter

            optim.zero_grad()
            loss.backward()
            optim.step()

            if math.isnan(loss.item()) == False:
                train_loss += loss.item()
                step_count += 1

            if (i + 1) % print_every == 0:
                running_loss += train_loss
                print('Epoch number {}, Step [{}/{}], Training Loss: {:.4f}, Training l1 Loss: {:.4f}, Training inter Loss: {:.4f}'
                      .format(t + 1, i + 1, len(train_loader), loss.item(), loss_l1.item(), loss_l1_inter.item()))
                train_loss = 0.0

                ################ validation
                model.eval()
                with torch.no_grad():
                    for X, y, y_m in validation_loader:
                        X = X.to(device)  # [N, SC, TC, H, W]
                        # X_m = X_m.to(device)  # [N, SC, H, W]
                        y = y.to(device)  # [N, H, W] with class indices (0, 1)
                        y_m = y_m.to(device)  # [N, SC, H, W]

                        X_spa, prediction = model(X)  # [N, H, W]

                        y = y.unsqueeze(1)

                        # intermediate loss
                        loss_l1_inter = criterion(X_spa, y_m)
                        # l1_inter = 0
                        # for kk in range(X_spa.shape[1]):
                        #     X_layer = X_spa[:, kk, :, :].unsqueeze(1)
                        #     l1_inter = l1_inter + criterion(X_layer, y)
                        #
                        # loss_l1_inter = l1_inter / X_spa.shape[1]

                        # L1 or L2 loss
                        loss_l1 = criterion(prediction, y)
                        # loss = loss_l1 + 100*loss_l1_inter
                        loss = loss_l1 + loss_l1_inter

                        if math.isnan(loss.item()) == False:
                            test_loss = loss.item()
                            test_step_count += 1

                        test_total_loss += test_loss
                        test_loss = 0

        test_loss_list.append(test_total_loss / test_step_count)
        train_loss_list.append(running_loss / step_count)

    model_name = 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\parameter_spatialM_temporalM_dualUnet_intLoss.pt'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.cpu().state_dict(), model_name)
    else:
        torch.save(model.cpu().state_dict(), model_name)

    ## Plotting batch-wise train loss curve:
    plt.plot(train_loss_list, '-o', label='train_loss', color='blue')
    plt.plot(test_loss_list, '-o', label='valid_loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('d:\\xhou4\\ML_CVR\\smooth8\\ML_data\\train\\valid_spatialM_temporalM_dualUnet_intLoss.png', dpi=300)
    plt.show()
