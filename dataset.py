import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import os
import functools
import json
import math
import cv2
import skvideo.io


## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class CNN3D_Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, inputs, labels, max_frames, spatial_transform=None, temporal_transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.inputs = inputs
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.max_frames = max_frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        input = self.inputs[index]

        # Load data
        print('Change directory {}'.format(os.path.dirname(input)))
        os.chdir(os.path.dirname(input))

        X = skvideo.io.vread(os.path.basename(input), outputdict={'-r': '1'})  # (frames, height, width, channel)
        X_list = []
        for i in range(X.shape[0]):
            X_list.append( self.spatial_transform(X[i]))

        X = torch.stack(X_list, dim=0)  # [frames * channels * height * weight]
        X = X.permute(1, 0, 2, 3)  # [channels * frames * height * weight]

        # The needed padding is the difference between the n_frames and MAX_NUM_FRAMES with zeros.
        p4d = (0, 0, 0, 0, 0, self.max_frames - X.shape[1])
        X = F.pad(X, p4d, "constant", 0)

        # y = torch.LongTensor([self.labels[index]])               # (labels) LongTensor are for int64 instead of FloatTensor
        y = self.labels[index]                                      # (label) clinical score
        return X, y
