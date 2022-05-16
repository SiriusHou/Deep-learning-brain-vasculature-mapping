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
from model_spatialM_unetA import unet_single_att_dsv_2D
from model_spatialM_unetA_LeeJunHyun import AttU_Net

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
                    if fnmatch.fnmatch(subfile, 'bold_spatialM_aal_[0-2]*'):
                        sub_filepath = os.path.join(data_root, subname, subfile)
                        self.input_IDs.append(sub_filepath)

                        parts = sub_filepath.split('/')
                        parts_prefix = '/'.join(parts[:-1])
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
        #         print(input_ID)

        # Load input
        img = nib.load(input_ID)
        train_image = img.get_fdata()
        train_image = train_image.astype(np.float32)
        #         img.uncache()

        # # For single channel
        # train_image = np.expand_dims(train_image, axis=2)

        # Transpose
        train_image = np.transpose(train_image, (2, 0, 1))

        A = torch.Tensor(train_image).type(torch.FloatTensor)

        # Select label
        label_ID = self.label_IDs[index]
        #         print(label_ID)

        # Load label
        taimg = nib.load(label_ID)
        label_image = taimg.get_fdata()
        label_image = label_image.astype(np.float32)
        #         taimg.uncache()

        B = torch.Tensor(label_image).type(torch.FloatTensor)
        return [A, B]



# %%

if __name__ == "__main__":

    validation_split = 0.05
    random_seed = 42
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    training_set = CVRDataset('/data1/xhou/ML_CVR/smooth8/ML_data/train')
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
    train_loader = data.DataLoader(training_set, batch_size=4, sampler=train_sampler, num_workers=10)
    validation_loader = data.DataLoader(training_set, batch_size=4, sampler=valid_sampler, num_workers=10)
    # model = unet_single_att_dsv_2D(in_channels=168, n_classes=1, feature_scale=1, is_batchnorm=False)
    model = AttU_Net(img_ch=168, output_ch=1)

    # single gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # #multiple gpu
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)  # enabling data parallelism

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.L1Loss()
    epochs = 100
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
            X = X[:, :-1, :, :]

            X = X.to(device)  # [N, C, H, W]
            y = y.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = np.squeeze(model(X))  # [N, H, W]
            loss = criterion(prediction, y)

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
                    for (X, y) in validation_loader:
                        X = X.to(device)  # [N, C, H, W]
                        y = y.to(device)  # [N, H, W] with class indices (0, 1)
                        prediction = np.squeeze(model(X))  # [N, H, W]
                        loss = criterion(prediction, y)

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
    plt.savefig('/data1/xhou/ML_CVR/smooth8/ML_data/train/UnetA_corr_gl_mBold_spatialM_aal_Epoch_100_valid_loss.png', dpi=300)
    plt.show()

    model_name = '/data1/xhou/ML_CVR/smooth8/ML_data/train/UnetA_corr_gl_mBold_spatialM_aal_Epoch_100.pt'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.cpu().state_dict(), model_name)
    else:
        torch.save(model.cpu().state_dict(), model_name)
