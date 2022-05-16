# %%
import time
import os
import glob
import pytorch_ssim
from torch.autograd import Variable
import fnmatch
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import nibabel as nib
import math
from corr_ChannelAttnunet_model import ChannelAttnUNet

# cwd = os.getcwd()
# print(cwd)

class CVRDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_root):
        'initialization'
        self.input_IDs = []
        self.label_IDs = []
        'Find the file location'
        for subname in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, subname)):
                for subfile in os.listdir(os.path.join(data_root, subname)):
                    if fnmatch.fnmatch(subfile, 'bold_cere_[0]*'):
                        sub_filepath = os.path.join(data_root, subname, subfile)
                        self.input_IDs.append(sub_filepath)

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
        #         img.uncache()

        #         train_image = np.pad(train_image, ((2, 3), (1, 2), (0, 0)), 'constant') #pad zero surrounding the image
        train_image = np.transpose(train_image, (2, 0, 1))

        A = torch.Tensor(train_image).type(torch.FloatTensor)

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
        return [A, B]


# %%

if __name__ == "__main__":

    validation_split = 0.2
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
    train_loader = data.DataLoader(training_set, batch_size=32, sampler=train_sampler, num_workers=10)
    validation_loader = data.DataLoader(training_set, batch_size=32, sampler=valid_sampler, num_workers=6)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ChannelAttnUNet(in_channels=50, n_classes=1, depth=5, padding=True, up_mode='upconv').to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss()
    ssim_loss = pytorch_ssim.SSIM()
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
            X = X.to(device)  # [N, C, H, W]
            y = y.to(device)  # [N, H, W] with class indices (0, 1)
            h, prediction = model(X)  # [N, H, W]
            #L1 or L2 loss
            prediction = np.squeeze(prediction)
            loss = criterion(y, prediction)

            # #SSIM loss
            # prediction = Variable(prediction, requires_grad=True)
            # y = y.unsqueeze(1)
            # loss = 1 - ssim_loss(y, prediction)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if math.isnan(loss.item()) == False:
                train_loss += loss.item()
                step_count += 1

            if (i + 1) % print_every == 0:
                running_loss += train_loss
                print('Epoch number {}, Step [{}/{}], Training Loss: {:.4f} '
                      .format(t + 1, i + 1, len(train_loader), loss.item()))
                train_loss = 0.0

                # validation
                model.eval()
                with torch.no_grad():
                    for X, y in validation_loader:
                        X = X.to(device)  # [N, C, H, W]
                        y = y.to(device)  # [N, H, W] with class indices (0, 1)
                        h, prediction = model(X)  # [N, H, W]
                        # L1 or L2 loss
                        prediction = np.squeeze(prediction)
                        loss = criterion(y, prediction)

                        # # SSIM loss
                        # y = y.unsqueeze(1)
                        # loss = 1 - ssim_loss(y, prediction)

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
    plt.savefig('d:\\xhou4\\ML_CVR\\smooth8\\ML_data\\train\\ChannelAttnUnet_avg3_ws_valid_loss.png', dpi=300)
    plt.show()

    torch.save(model.state_dict(), 'd:\\xhou4\\ML_CVR\\smooth8\\ML_data\\ChannelAttnUnet_avg3_WS_134sub.pt')
