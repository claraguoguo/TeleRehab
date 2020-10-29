
from models.cnn3d import CNN3D


def generate_model(model_name, config):
    assert model_name in ['cnn']
    if model_name == 'cnn':
        frame_size = config.getint(model_name, 'frame_size')
        model = CNN3D(t_dim=45, img_x=frame_size, img_y=frame_size)
        print("generating CNN3D model")

    return model
