
from models.cnn3d import CNN3D


def generate_model(model_name, max_num_frame, config):
    assert model_name in ['cnn']
    if model_name == 'cnn':
        frame_size = config.getint(model_name, 'frame_size')
        model = CNN3D(t_dim=max_num_frame, img_x=frame_size, img_y=frame_size)
        print("generating CNN3D model")

    return model
