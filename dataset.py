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

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, n_frames, image_loader):
    video = []
    assert(n_frames != 0)
    for i in range(1, n_frames+1):
        image_path = os.path.join(video_dir_path, 'image_{:04d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(video_path, max_duration):
    dataset = []

    n_frames = len(os.listdir(video_path))
    # begin_t = 1
    # end_t = n_frames
    sample = {
        'video': video_path,
        'n_frames': n_frames,
        'frame_indices': list(range(1, 1 + n_frames))
    }
    return sample
    # sample_i = copy.deepcopy(sample)
    # sample_i['frame_indices'] = list(range(i, i + sample_duration))
    # sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
    # dataset.append(sample_i)

    # step = sample_duration
    # for i in range(1, (n_frames - sample_duration + 1), step):
    #     sample_i = copy.deepcopy(sample)
    #     sample_i['frame_indices'] = list(range(i, i + sample_duration))
    #     sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
    #     dataset.append(sample_i)

    # return dataset

#
# class Video(data.Dataset):
#     def __init__(self, video_path, clinical_score,
#                  spatial_transform=None, temporal_transform=None,
#                  , get_loader=get_default_video_loader):
#         self.data = make_dataset(video_path, max_duration),
#         self.label = clinical_score,
#         self.spatial_transform = spatial_transform
#         self.temporal_transform = temporal_transform
#         self.loader = get_loader()
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         path = self.data[index]['video']
#         n_frames = self.data[index]['n_frames']
#         frame_indices = self.data[index]['frame_indices']
#         if self.temporal_transform is not None:
#             frame_indices = self.temporal_transform(frame_indices)
#         clip = self.loader(path, n_frames)
#         if self.spatial_transform is not None:
#             clip = [self.spatial_transform(img) for img in clip]
#         clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
#
#         target = self.label[index]
#
#         return clip, target
#
#     def __len__(self):
#         return len(self.data)
#
#
#
#
#
#
#










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

    def _extract_frames_from_video(self, video_path):
        X = []
        n_frames = 0
        cap = cv2.VideoCapture(video_path)  # capturing the video from the given path
        frameRate = cap.get(5)  # frame rate
        while (cap.isOpened()):
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):

                if self.temporal_transform is not None:
                    # TODO
                    continue
                if self.spatial_transform is not None:
                    frame = self.spatial_transform(frame)

                X.append(frame.squeeze_(0))
                n_frames += 1

        cap.release()

        X = torch.stack(X, dim=0)  # [frames * channels * height * weight]
        X = X.permute(1, 0, 2, 3)  # [channels * frames * height * weight]

        # The needed padding is the difference between the
        # n_frames and MAX_NUM_FRAMES with zeros.
        p4d = (0, 0, 0, 0, 0, self.max_frames - n_frames)
        out = F.pad(X, p4d, "constant", 0)
        # padded_img = F.pad(image, [0, MAX_FRAMES - img.size(2), 0, MAX_FRAMES - img.size(1)])
        return out

    # def read_images(self, input):
    #     X = []
    #     n_frames = len(os.listdir(os.path.join(path, selected_folder)))
    #
    #     # TODO: check what is the max number of frames for the target exercise
    #     MAX_NUM_FRAMES = 45
    #
    #     assert(n_frames != 0)
    #     for i in range(1, n_frames + 1):
    #         image_name = os.path.join(path, selected_folder, 'image_{:04d}.jpg'.format(i))
    #
    #         image = Image.open(image_name).convert('RGB')
    #
    #         if self.temporal_transform is not None:
    #             # TODO
    #             continue
    #         if self.spatial_transform is not None:
    #             image = self.spatial_transform(image)
    #         X.append(image.squeeze_(0))
    #
    #     X = torch.stack(X, dim=0)           # [frames * channels * height * weight]
    #     X = X.permute(1, 0, 2, 3)           # [channels * frames * height * weight]
    #
    #     # The needed padding is the difference between the
    #     # n_frames and MAX_NUM_FRAMES with zeros.
    #     p4d = (0, 0, 0, 0, 0, MAX_NUM_FRAMES - n_frames)
    #     out = F.pad(X, p4d, "constant", 0)
    #     # padded_img = F.pad(image, [0, MAX_FRAMES - img.size(2), 0, MAX_FRAMES - img.size(1)])
    #     return out

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        input = self.inputs[index]

        # Load data
        # X = self._extract_frames_from_video(input)  # (input) spatial images
        X = skvideo.io.vread(input, outputdict={'-r': '1'})

        X = torch.from_numpy(X)  # [ frames * height * weight * channels]
        X = X.permute(3, 0, 1, 2)  # [channels * frames * height * weight]

        # The needed padding is the difference between the
        # n_frames and MAX_NUM_FRAMES with zeros.
        p4d = (0, 0, 0, 0, 0, self.max_frames - X.shape[1])
        X = F.pad(X, p4d, "constant", 0)
        # padded_img = F.pad(image, [0, MAX_FRAMES - img.size(2), 0, MAX_FRAMES - img.size(1)])

        # y = torch.LongTensor([self.labels[index]])               # (labels) LongTensor are for int64 instead of FloatTensor
        y = self.labels[index]                                      # (label) clinical score
        # print(X.shape)
        return X, y
