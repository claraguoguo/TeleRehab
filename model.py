
from models.model_names import *
import torchvision.models as models
import torch


def generate_model(model_name, max_num_frame, config):
    assert model_name in ['cnn', 'c3d', 'resnet', 'binary_cnn', 'binary_resnet']
    frame_size = config.getint(model_name, 'frame_size')
    if model_name == 'cnn':
        model = CNN3D(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("Loading CNN3D model")

    elif model_name =='c3d':
        model = C3D(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("Loading c3d model")

    elif model_name =='resnet':
        model = models.video.r3d_18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)
        print("Loading r3d_18 model")

    elif model_name == 'binary_cnn':
        model = Binary_CNN3d(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("Loading binary_cnn model")

    elif model_name == 'binary_resnet':
        model = models.video.r3d_18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        print("Loading binary_resnet (r3d_18 pre-trained) model")

    else:
        print('Invalid model name')
        assert False

    return model
