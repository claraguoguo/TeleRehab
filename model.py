
from models.cnn3d import CNN3D
import torchvision.models as models


def generate_model(model_name, max_num_frame, config):
    assert model_name in ['cnn', 'resnet']
    if model_name == 'cnn':
        frame_size = config.getint(model_name, 'frame_size')
        model = CNN3D(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("Loading CNN3D model")

    elif model_name =='resnet':
        model = models.video.r3d_18(pretrained=True)
        print("Loading r3d_18 model")

    else:
        print('Invalid model name')
        assert False

    return model
