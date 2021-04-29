
from models.model_names import *
import torchvision.models as models
import torch
import json

def generate_model(model_name, max_num_frame, config):
    if model_name == 'cnn':
        frame_size = config.getint(model_name, 'frame_size')
        model = CNN3D(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("Loading CNN3D model")

    elif model_name == 'c3d':
        frame_size = config.getint(model_name, 'frame_size')
        model = C3D(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("Loading c3d model")

    elif model_name == 'resnet':
        model = models.video.r3d_18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)
        print("Loading r3d_18 model")

    elif model_name == 'binary_cnn':
        frame_size = config.getint(model_name, 'frame_size')
        model = Binary_CNN3d(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("Loading binary_cnn model")

    elif model_name == 'binary_resnet':
        model = models.video.r3d_18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        print("Loading binary_resnet (r3d_18 pre-trained) model")

    elif model_name == 'lstm':
        hidden_dim = config.getint(model_name, 'n_hidden')
        n_joints = config.getint(model_name, 'n_joints')
        n_categories = config.getint(model_name, 'n_categories')
        n_layer = config.getint(model_name, 'n_layer')
        n_features = len(json.loads(config.get(model_name, 'feat_indices')))
        should_use_features = config.getint(model_name, 'should_use_features')
        if(should_use_features):
            model = LSTM(n_features, hidden_dim, n_categories, n_layer, max_num_frame)
        else:
            model = LSTM(n_joints, hidden_dim, n_categories, n_layer, max_num_frame)
        print("Loading LSTM model")

    elif model_name == 'mlp':
        n_repetition = config.getint('dataset', 'n_repetition')
        n_features = len(json.loads(config.get(model_name, 'feat_indices')))
        input_dim = n_features * n_repetition
        model = MLP(input_dim)
        print("Loading MLP model")

    elif model_name == 'linearReg':
        n_repetition = config.getint('dataset', 'n_repetition')
        n_features = len(json.loads(config.get(model_name, 'feat_indices')))
        input_dim = n_features * n_repetition
        model = LinearReg(input_dim)
        print("Loading linearReg model")
    else:
        print('Invalid model name')
        assert False

    return model
