import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import skvideo.io
import numpy as np
from sklearn.model_selection import train_test_split

from model import generate_model
from opts import parse_opts

## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class CNN3D_Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, config, inputs, labels, max_frames, spatial_transform=None, temporal_transform=None):
        "Initialization"
        self.config = config
        self.labels = labels
        self.inputs = inputs
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.max_frames = max_frames
        # Note: fps MUST be s String, b/c skvideo.io.vread is expecting a string input.
        self.fps = config.get('dataset', 'fps')

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        input = self.inputs[index]

        # Load data
        X = skvideo.io.vread(input, outputdict={'-r': self.fps})  # (frames, height, width, channel)
        X_list = []
        for i in range(X.shape[0]):
            X_list.append(self.spatial_transform(X[i]))

        X = torch.stack(X_list, dim=0)  # [frames * channels * height * weight]
        X = X.permute(1, 0, 2, 3)  # [channels * frames * height * weight]

        # The needed padding is the difference between the n_frames and MAX_NUM_FRAMES with zeros.
        p4d = (0, 0, 0, 0, 0, self.max_frames - X.shape[1])
        X = F.pad(X, p4d, "constant", 0)

        # y = torch.LongTensor([self.labels[index]])               # (labels) LongTensor are for int64 instead of FloatTensor
        y = self.labels[index]                                     # (label) clinical score
        return X, y


class Weighted_Loss_Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, config, inputs, labels, max_frames, spatial_transform=None, temporal_transform=None, weights=None):
        "Initialization"
        self.config = config
        self.labels = labels
        self.inputs = inputs
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.weights = weights
        self.max_frames = max_frames
        # Note: fps MUST be s String, b/c skvideo.io.vread is expecting a string input.
        self.fps = config.get('dataset', 'fps')

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        input = self.inputs[index]

        # Load data
        X = skvideo.io.vread(input, outputdict={'-r': self.fps})  # (frames, height, width, channel)
        X_list = []
        for i in range(X.shape[0]):
            X_list.append(self.spatial_transform(X[i]))

        X = torch.stack(X_list, dim=0)  # [frames * channels * height * weight]
        X = X.permute(1, 0, 2, 3)  # [channels * frames * height * weight]

        # The needed padding is the difference between the n_frames and MAX_NUM_FRAMES with zeros.
        p4d = (0, 0, 0, 0, 0, self.max_frames - X.shape[1])
        X = F.pad(X, p4d, "constant", 0)

        # y = torch.LongTensor([self.labels[index]])               # (labels) LongTensor are for int64 instead of FloatTensor
        y = self.labels[index]                                     # (label) clinical score

        # Get the corresponding weight for the current class
        w = self.weights[int(y)]
        return X, y, w



## ---------------------- LSTM Dataloader ---------------------- ##

class LSTM_Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, config, inputs, labels, max_frames):
        "Initialization"
        self.config = config
        self.labels = labels
        self.inputs = inputs
        self.max_frames = max_frames
        self.skeletal_data_path = config.get('dataset', 'skeletal_data_path')

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        input = self.inputs[index]

        os.chdir(self.skeletal_data_path)
        # Load data from the text file
        txt_file_name = os.path.join(*(input.split('/')[-6:])).replace("/", "_").split(".")[0]

        data = np.loadtxt(txt_file_name + ".txt", delimiter=',')

        # Convert numpy array to tensor
        X = torch.from_numpy(data)

        X_len = X.shape[0]         # Number of frames

        # The needed padding is the difference between the X.shape[0] and max_frames.
        p1d = (0, 0, 0, self.max_frames - X_len)
        X = F.pad(X, p1d, "constant", 0)                           # [max_frames, n_joints]

        y = torch.LongTensor([self.labels[index]])               # (labels) LongTensor are for int64 instead of FloatTensor

        # label = self.labels[index]                                 # (label) clinical score
        # y = torch.tensor(())
        # if (label == 1):
        #     y = y.new_ones((X.shape[0], 1), dtype=torch.long)            # [max_frames, 1]
        # else:
        #     y = y.new_zeros((X.shape[0], 1), dtype=torch.long)           # [max_frames, 1]
        return X, y, X_len


# LSTM_Dataset_Wrapper handles errors from LSTM_Dataset
class LSTM_Dataset_Wrapper(LSTM_Dataset):
    __init__ = LSTM_Dataset.__init__
    def __getitem__(self, index):
        try:
            return super(LSTM_Dataset, self).__getitem__(index)
        except Exception as e:
            print(e)



class MLP_Dataset(data.Dataset):

    def __init__(self, X, y, config):
        self.features = X
        self.label = y
        self.skeletal_features_path = config.get('dataset', 'skeletal_features_path')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]

        # Load data from the text file
        txt_file_name = os.path.join(*(features.split('/')[-6:])).replace("/", "_").split(".")[0]

        if (txt_file_name[-1] == "_"):
            # TODO: need a better fix
            # Naming is inconsistent in KIMORE, some videos has an extra underscore. The extra underscore needs to be removed
            # i.e. 'CG_Expert_E_ID9_Es1_rgb_Blur_rgb271114_123334_'
            txt_file_name = txt_file_name[:-1]

        data = np.loadtxt(os.path.join(self.skeletal_features_path, txt_file_name + ".txt"), delimiter=',')

        data = data[:, :3]
        # Convert numpy array to tensor
        X = torch.from_numpy(data)
        # Flatten X
        X = X.flatten()
        label = self.label[index]
        return X, label