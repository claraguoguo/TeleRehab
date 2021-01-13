import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import os
from dataset import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import configparser

# from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
# from temporal_transforms import LoopPadding

def plot_confusion_matrix(cm, auc, model_name, config):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot()
    disp.ax_.set_title("Confusion Matrix")
    
    epoch = config.getint(model_name, 'epoch')
    plt.savefig("cm_auc_{}_epoch_{}.png".format(auc, epoch))

def show_binary_classifier_metrics(y_true, y_pred, model_name, config):
    print('\n Binary Classifier Metrics Results')
    print('Total number of test cases: {}'.format(len(y_true)))

    y_true = [int(a) for a in y_true]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2 * tp / (2 * tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr + tnr) / 2
    # MCC
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Save metrics results into a text file
    with open("Metrics_Results.txt", "w") as text_file:
        print(f"Labels:  {y_true} \nOutputs: {y_pred} \n", file=text_file)
        print(f"True positive: {tp}", file=text_file)
        print(f"False positive: {fp}", file=text_file)
        print(f"True negative: {tn}", file=text_file)
        print(f"False negative: {fn}", file=text_file)

        print(f"True positive rate (recall): {tpr}", file=text_file)
        print(f"False positive rate: {fpr}", file=text_file)
        print(f"Precision: {precision}", file=text_file)
        print(f"True negative rate: {tnr}", file=text_file)
        print(f"F1: {f1}", file=text_file)
        print(f"MCC: {mcc}", file=text_file)
        print(f"ROC-AUC: {auc}", file=text_file)

        print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']), file=text_file)

    # generate confusion matrix figure
    plot_confusion_matrix(cm, auc, model_name, config)

def plot_labels_and_outputs(labels, outputs, config, model_name):

    epoch = config.getint(model_name, 'epoch')
    lr = config.getfloat(model_name, 'lr')
    bs = config.getint(model_name, 'batch_size')
    loss_fn = config.get(model_name, 'loss')
    fps = config.get('dataset', 'fps')

    x = np.arange(0, len(labels), 1)
    plt.plot(x, labels, 'o', color='black', label='Labels')
    plt.plot(x, outputs, 'o', color='red', label='Predictions')
    plt.ylabel("Score")
    plt.xlabel("Test Data")
    plt.legend(loc='best')
    plt.title("Scatterplot of Labels and Predictions \n {0}_{1}_lr{2}_epoch{3}_bs{4}_fps{5}.png".format(
        model_name, loss_fn , lr, epoch, bs, fps))
    plt.savefig("{0}_test_scatterplot_{1}_lr{2}_epoch{3}_bs{4}_fps{5}.png".format(
        model_name, loss_fn , lr, epoch, bs, fps))

def change_dir(new_dir):
    print('Change directory to{}'.format(new_dir))
    os.chdir(new_dir)

def get_config(config_path):
    if not os.path.exists(config_path):
        raise FileExistsError('config file does not exist')
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def get_model_name(config):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_"
    path += "epoch{}_".format(config["num_epochs"])
    path += "bs{}_".format(config["batch_size"])
    path += "lr{}".format(config["learning_rate"])

    return path


def get_relevant_indices(dataset, classes, target_classes):
    """ Returns the indices for datapoints in the dataset that
    belongs to the desired target classes, a subset of all possible classes.

    Args:
        dataset: Dataset object
        classes: A list of strings denoting the name of each class
        target_classes: A list of strings denoting the name of the desired classes.
                        Should be a subset of the 'classes'
    Returns:
        indices: list of indices that have labels corresponding to one of the target classes
    """
    indices = []
    for i in range(len(dataset)):
        # Check if the label is in the target classes
        label_index = dataset[i][1] # ex: 3
        label_class = classes[label_index] # ex: 'cat'
        if label_class in target_classes:
            indices.append(i)

    return indices


def normalize_label(labels):
    """
    Given a tensor containing 2 possible values, normalize this to 0/1

    Args:
        labels: a 1D tensor containing two possible scalar values
    Returns:
        A tensor normalize to 0/1 value
    """
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    norm_labels = (labels - min_val)/(max_val - min_val)

    return norm_labels



################################### LOADING DATA ################################### 
def get_data_loader(train_list, test_list, train_label, test_label, model_name, max_frames, config):
    # Use the mean and std 
    # https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    frame_size = config.getint(model_name, 'frame_size')
    ## sample_duration will be the number of frames = duration of video in seconds
    # opt.sample_duration = len(os.listdir("tmp"))
    # TODO: Normalize Image (center / min-max) & Map rgb --> [0, 1]
    spatial_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.CenterCrop(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    ## temporal_transform = LoopPadding(opt.sample_duration)
    temporal_transform = None

    batch_size = config.getint(model_name, 'batch_size')
    n_threads = config.getint(model_name, 'n_threads')

    train_set = CNN3D_Dataset(config, train_list, train_label, max_frames, spatial_transform=spatial_transform,
                              temporal_transform=temporal_transform)
    valid_set = CNN3D_Dataset(config, test_list, test_label, max_frames, spatial_transform=spatial_transform,
                              temporal_transform=temporal_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                                            shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=True),
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
    #                                            shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=False, num_workers=n_threads, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                              shuffle=False, num_workers=n_threads, pin_memory=True)
    return train_loader, valid_loader


################################### LOADING DATA with Weighted Loss################################### 
def get_weighted_loss_data_loader(train_list, test_list, train_label, test_label, model_name, max_frames, config):
    # Use the mean and std 
    # https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    frame_size = config.getint(model_name, 'frame_size')
    ## sample_duration will be the number of frames = duration of video in seconds
    # opt.sample_duration = len(os.listdir("tmp"))
    # TODO: Normalize Image (center / min-max) & Map rgb --> [0, 1]
    spatial_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.CenterCrop(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    ## temporal_transform = LoopPadding(opt.sample_duration)
    temporal_transform = None

    batch_size = config.getint(model_name, 'batch_size')
    n_threads = config.getint(model_name, 'n_threads')

    train_set = Weighted_Loss_Dataset(config, train_list, train_label, max_frames, spatial_transform=spatial_transform,
                              temporal_transform=temporal_transform)
    valid_set = Weighted_Loss_Dataset(config, test_list, test_label, max_frames, spatial_transform=spatial_transform,
                              temporal_transform=temporal_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                                            shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=True),
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
    #                                            shuffle=False, num_workers=n_threads, pin_memory=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=False, num_workers=n_threads, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                              shuffle=False, num_workers=n_threads, pin_memory=True)
    return train_loader, valid_loader
