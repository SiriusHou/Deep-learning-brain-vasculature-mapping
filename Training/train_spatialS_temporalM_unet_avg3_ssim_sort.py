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
from model_spatialS_temporalM_unet_avg3_ssim_sort import ChannelAttnUNet

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
                    if fnmatch.fnmatch(subfile, 'bold_cere_sort_[0]*'):
                        sub_filepath = os.path.join(data_root, subname, subfile)
                        self.input_IDs.append(sub_filepath)

                        # sub_filepath = os.path.join(data_root, subname, subfile)
                        # self.input_IDs.append(sub_filepath)

                        parts = sub_filepath.split('\\')
                        parts_prefix = '\\'.join(parts[:-1])
                        parts_end = parts[-1][-7:]

                        self.minput_IDs.append(os.path.join(parts_prefix, 'bold_mean_rp_' + parts_end))
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
        #         img.uncache()

        #         train_image = np.pad(train_image, ((2, 3), (1, 2), (0, 0)), 'constant') #pad zero surrounding the image
        train_image = np.transpose(train_image, (2, 0, 1))

        A = torch.Tensor(train_image).type(torch.FloatTensor)

        # Select mean input
        minput_ID = self.minput_IDs[index]
        # print(input_ID)

        # Load mean input
        mimg = nib.load(minput_ID)
        train_mimage = mimg.get_fdata()
        train_mimage = train_mimage.astype(np.float32)
        #         img.uncache()

        A_m = torch.Tensor(train_mimage).type(torch.FloatTensor)
        A_m = A_m.unsqueeze(0)

        # Select label
        label_ID = self.label_IDs[index]
        # print(label_ID)

        # Load label
        taimg = nib.load(label_ID)
        label_image = taimg.get_fdata()
        label_image = label_image.astype(np.float32)
        #         taimg.uncache()

        #         label_image = np.pad(label_image, ((2, 3), (1, 2)), 'constant') #pad zero surrounding the image

        B = torch.Tensor(label_image).type(torch.FloatTensor)
        return [A, A_m, B]


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
    train_loader = data.DataLoader(training_set, batch_size=64, sampler=train_sampler, num_workers=10)
    validation_loader = data.DataLoader(training_set, batch_size=64, sampler=valid_sampler, num_workers=6)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ChannelAttnUNet(in_channels=50, n_classes=1, depth=5, padding=True, up_mode='upconv').to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss()
    criterion_cc = nn.CrossEntropyLoss()
    epochs = 50
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

        for i, (X, X_m, y) in enumerate(train_loader):
            X = X.to(device)  # [N, C, H, W]
            X_m = X_m.to(device) # [N, 1, H, W]
            y = y.to(device)  # [N, H, W] with class indices (0, 1)
            h, x_h, prediction = model(X, X_m)  # [N, H, W]

            y = y.unsqueeze(1)
            # ssim_index_all = np.zeros(X.shape[0:2])
            ssim_output = []
            for kk in range(h.shape[1]):

                X_layer = X[:, kk, :, :].unsqueeze(1)
                ssim_index = ssim(y, X_layer, window_size=25, radius=4, size_average=None, val_range=10)
                ssim_index = ssim_index.unsqueeze(1)
                ssim_output.append(ssim_index)

            ssim_index_all = torch.cat(ssim_output, dim=1)
            # ssim_index_all = F.softmax(ssim_index_all, dim=1)

            # ssim_index_max = torch.argmax(ssim_index_all, 1, keepdim=True)
            # ssim_index_max_all = torch.zeros_like(ssim_index_all)
            # ssim_index_max_all.scatter_(1, ssim_index_max, 1)
            # ssim_index_max_all = ssim_index_max_all.long()

            loss_ssim = criterion(h, ssim_index_all)

            #SSIM loss
            # loss_ssim = 1 - ssim(y, x_h, val_range=10)
            # loss_ssim = 1 - msssim(y, x_h, val_range=10, normalize="relu")

            #L1 or L2 loss
            loss_l1 = criterion(prediction, y)
            loss = loss_ssim + loss_l1

            optim.zero_grad()
            loss.backward()
            optim.step()

            if math.isnan(loss.item()) == False:
                train_loss += loss.item()
                step_count += 1

            if (i + 1) % print_every == 0:
                running_loss += train_loss
                print('Epoch number {}, Step [{}/{}], Training Loss: {:.4f}, Training ssim Loss: {:.4f}, Training l1 Loss: {:.4f} '
                      .format(t + 1, i + 1, len(train_loader), loss.item(), loss_ssim.item(), loss_l1.item()))
                train_loss = 0.0

                # validation
                model.eval()
                with torch.no_grad():
                    for X, X_m, y in validation_loader:
                        X = X.to(device)  # [N, C, H, W]
                        X_m = X_m.to(device)  # [N, 1, H, W]
                        y = y.to(device)  # [N, H, W] with class indices (0, 1)
                        h, x_h, prediction = model(X, X_m)  # [N, H, W]

                        y = y.unsqueeze(1)
                        ssim_output = []
                        for kk in range(h.shape[1]):
                            X_layer = X[:, kk, :, :].unsqueeze(1)
                            ssim_index = ssim(y, X_layer, window_size=25, radius=4, size_average=None, val_range=10)
                            ssim_index = ssim_index.unsqueeze(1)
                            ssim_output.append(ssim_index)

                        ssim_index_all = torch.cat(ssim_output, dim=1)
                        # ssim_index_all = F.softmax(ssim_index_all, dim=1)

                        loss_ssim = criterion(h, ssim_index_all)

                        # SSIM loss
                        # loss_ssim = 1 - ssim(y, x_h, val_range=10)
                        # loss_ssim = 1 - msssim(y, x_h, val_range=10, normalize="relu")

                        # L1 or L2 loss
                        loss_l1 = criterion(prediction, y)
                        loss = loss_ssim + loss_l1

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
    plt.savefig('d:\\xhou4\\ML_CVR\\smooth8\\ML_data\\train\\valid_spatialS_temporalM_unet_avg3_ssim_sort.png', dpi=300)
    plt.show()

    torch.save(model.state_dict(), 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\parameter_spatialS_temporalM_unet_avg3_ssim_sort1.pt')
