import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import os
from dataset import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import configparser
from sklearn import metrics
import math
import glob
from scipy import stats
import pandas as pd

def getSampler(labels, config):
    binary_threshold = config.getint('dataset', 'binary_threshold')
    unique_labels = np.sort(np.unique(labels))
    class_counts = [(labels[labels == label]).shape[0] for label in unique_labels]
    class_weights = [sum(class_counts) / class_counts[i] for i in range(len(class_counts))]
    weights = torch.ones(labels.shape)
    weights[labels == 0] = class_weights[0]
    weights[labels == 1] = class_weights[1]
    weights = torch.FloatTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler

def plot_confusion_matrix(cm, auc, model_name, config):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['class 0', 'class 1'])
    disp = disp.plot()
    disp.ax_.set_title("Confusion Matrix")
    
    epoch = config.getint(model_name, 'epoch')
    plt.savefig("cm_auc_{}_epoch_{}.png".format(auc, epoch))

def record_test_results(output_path, test_ID, labels_list, predict_list, test_loss, model_name, config,
                        cv_scores_mse=[], cv_scores_spearman=[]):
    # Save metrics results into a text file
    content = ''
    file_name = os.path.join(output_path, "Test_Results.txt")

    # Compute spearman correlation and p-value
    rho, pval = stats.spearmanr(predict_list, labels_list)

    should_use_features = model_name not in ['cnn', 'resnet', 'c3d']

    if should_use_features:
        feature_dir = os.path.dirname(config.get('dataset', 'skeletal_features_path'))
        feature_info_path = os.path.join(feature_dir, 'features_info.txt')
        with open(feature_info_path) as f:
            feature = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        features = [x.strip() for x in feature]

        feat_indices = json.loads(config.get(model_name, 'feat_indices'))
        selected_features = [features[index] for index in feat_indices]

    with open(file_name, "w") as text_file:
        print(content, file=text_file)
        print('Test IDs: ' + str(test_ID), file=text_file)
        print('Test labels_list: ' + str(list(np.around(np.array(labels_list), 2))), file=text_file)
        print('Test predicts_list:' + str(list(np.around(np.array(predict_list), 2))), file=text_file)

        print("Test loss: {:0.4f}".format(test_loss), file=text_file)

        print('Spearman correlation coefficient: {0:0.4f} with p-value: {1:0.4f}'.format(rho, pval), file=text_file)
        if should_use_features:
            print("Selected skeletal features: " + str(selected_features), file=text_file)

        if len(cv_scores_mse) != 0:
            print("5-fold Cross Validation Scores (with MSE): {} mean: {:0.2f}".format(
                cv_scores_mse,cv_scores_mse.mean()), file=text_file)
        if len(cv_scores_spearman) != 0:
            print("5-fold Cross Validation Scores (with Spearman): {} mean: {:0.2f}".format(
                cv_scores_spearman, cv_scores_spearman.mean()), file=text_file)

def write_binary_classifier_metrics(y_true, y_pred, y_pred_prob, y_IDs, model_name, config):
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
    # True negative rate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2 * tp / (2 * tp + fp + fn)
    # ROC-AUC for binary classification
    roc_auc = metrics.roc_auc_score(y_true, y_pred_prob)
    roc_auc = round(roc_auc, 2)
    # MCC
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Save metrics results into a text file
    content = ''
    file_name = "Metrics_Results.txt"

    should_use_weighted_loss = config.getint(model_name, 'should_use_weighted_loss')
    if should_use_weighted_loss:
        content = 'With Weighted Loss: \n'

    with open(file_name, "w") as text_file:
        print(content, file=text_file)
        print(f"Test IDs:  {y_IDs} \n", file=text_file)
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
        print(f"ROC-AUC: {roc_auc}", file=text_file)

        print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']), file=text_file)

    # generate confusion matrix figure
    plot_confusion_matrix(cm, roc_auc, model_name, config)

def plot_training_loss(model_name, type, train_data, test_loss, config, output_path):
    """
    Plot the training loss/error curve given the data from CSV
    """
    epoch = config.getint(model_name, 'epoch')
    lr = config.getfloat(model_name, 'lr')
    bs = config.getint(model_name, 'batch_size')
    loss_fn = config.get(model_name, 'loss')
    plt.figure()
    plt.title("{0} over training epochs \n {1}_lr{2}_epoch{3}_bs{4}_test{5:.3f}".format(
        type, model_name, lr, epoch, bs, test_loss))
    plt.plot(np.arange(1, epoch + 1), train_data, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel(loss_fn + type)
    plt.legend(loc='best')
    plt.savefig("{7}/{0}_{1}_{2}_lr{3}_epoch{4}_bs{5}_test{6:.3f}.png".format(
        model_name, type, loss_fn, lr, epoch, bs, test_loss, output_path))
    plt.close()


def plot_labels_and_outputs(labels, outputs, config, model_name, ids, test_loss, plot_name=''):
    plt.figure()
    epoch = config.getint(model_name, 'epoch')
    lr = config.getfloat(model_name, 'lr')
    bs = config.getint(model_name, 'batch_size')
    loss_fn = config.get(model_name, 'loss')
    fps = config.get('dataset', 'fps')

    # Compute spearman correlation and p-value
    rho, _ = stats.spearmanr(outputs, labels)

    x = np.arange(0, len(labels), 1)
    plt.plot(x, labels, 'o', color='black', label='Actual')
    plt.plot(x, outputs, 'o', color='red', label='Predicted')

    # for i in range(len(labels)):
    #     label = "{}".format(ids[i])
    #
    #     plt.annotate(label,  # this is the text
    #                 (x[i], labels[i]),  # this is the point to label
    #                 textcoords="offset points",  # how to position the text
    #                 xytext=(0, 10),  # distance from text to points (x,y)
    #                 ha='center')  # horizontal alignment can be left, right or center

    plt.ylabel("Score")
    plt.xlabel("Test Data")
    plt.legend(loc='best')
    plt.xticks(x, ids, fontsize=8, rotation=45)

    if (model_name not in ['cnn', 'resnet', 'c3d']):
        # 3D-CNN models' config section do not have 'feat_indices'
        feat_indices = json.loads(config.get(model_name, 'feat_indices'))

        plt.title("Scatter plot of Actual v.s. Predicted Scores "
                  "\nTest loss: {6:0.2f} Spearman Corr: {7:0.2f} Features_Ind:{8}"
                  "\n{0}_{1}_lr{2}_epoch{3}_bs{4}_fps{5}".format(
            model_name, loss_fn, lr, epoch, bs, fps, test_loss, rho, feat_indices), fontsize=10)
    else:
        plt.title("Scatter plot of Actual v.s. Predicted Scores "
                  "\nTest loss: {6:0.2f} Spearman Corr: {7:0.2f}"
                  "\n{0}_{1}_lr{2}_epoch{3}_bs{4}_fps{5}".format(
            model_name, loss_fn, lr, epoch, bs, fps, test_loss, rho), fontsize=10)


    if not plot_name:
        plot_name = "{0}_{1}_{6:0.1f}_spearman_{7:0.1f}_lr{2}_epoch{3}_bs{4}_fps{5}.png".format(
            model_name, loss_fn, lr, epoch, bs, fps, test_loss, rho)
    plt.savefig(plot_name)
    plt.close()

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

def get_data_loader(train_list, test_list, train_label, test_label, model_name, max_frames, config ):
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
        transforms.CenterCrop((540, 500)),
        transforms.Resize((frame_size, frame_size)),
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=False, num_workers=n_threads, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                              shuffle=False, num_workers=n_threads, pin_memory=True)
    return train_loader, valid_loader



def get_weighted_loss_data_loader(train_list, test_list, train_label, test_label,
                                  model_name, max_frames, config, label_to_weights):
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
        transforms.CenterCrop((540, 500)),
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    ## temporal_transform = LoopPadding(opt.sample_duration)
    temporal_transform = None

    batch_size = config.getint(model_name, 'batch_size')
    n_threads = config.getint(model_name, 'n_threads')
    train_set = Weighted_Loss_Dataset(config, train_list, train_label, max_frames, spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform, weights=label_to_weights)
    test_set = Weighted_Loss_Dataset(config, test_list, test_label, max_frames, spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform, weights=label_to_weights)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_threads,
                                               pin_memory=True, sampler=getSampler(train_label, config))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=n_threads,
                                               pin_memory=True)
    return train_loader, test_loader

def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_nn_data_loader(train_list, test_list, train_label, test_label,
                                  model_name, config):
    batch_size = config.getint(model_name, 'batch_size')
    train_dataset = NN_Dataset(train_list, train_label, config, model_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = NN_Dataset(test_list, test_label, config, model_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def get_lstm_skeletal_features_data_loader(train_list, test_list, train_label, test_label,
                                  model_name, max_frames, config):

    batch_size = config.getint(model_name, 'batch_size')
    n_threads = config.getint(model_name, 'n_threads')
    train_set = LSTM_Skeletal_Features_Dataset(config, train_list, train_label, max_frames, model_name)
    test_set = LSTM_Skeletal_Features_Dataset(config, test_list, test_label, max_frames, model_name)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=n_threads, collate_fn=my_collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=n_threads, collate_fn=my_collate)

    return train_loader, test_loader


def get_lstm_data_loader(train_list, test_list, train_label, test_label,
                                  model_name, max_frames, config):

    batch_size = config.getint(model_name, 'batch_size')
    n_threads = config.getint(model_name, 'n_threads')
    train_set = LSTM_Dataset(config, train_list, train_label, max_frames)
    test_set = LSTM_Dataset(config, test_list, test_label, max_frames)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=n_threads, collate_fn=my_collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              num_workers=n_threads, collate_fn=my_collate)

    return train_loader, test_loader


# Get the max line counts in all the .txt files under given file_root
def get_max_line_counts(file_root):
    max_lines = 0
    file_count = 0
    for filepath in glob.glob(file_root + '/*.txt', recursive=True):
        num_lines = sum(1 for line in open(filepath))
        if (num_lines > max_lines):
            max_lines = num_lines

        file_count += 1
    print("Iterate through {0} files, max line counts: {1}".format(file_count, max_lines))
    return max_lines

def get_fixed_test_data(all_X_list, all_y_list):
    fixed_colab_test_ID = ['B_ID5', 'NE_ID6', 'P_ID6', 'B_ID1', 'S_ID9', 'NE_ID13', 'P_ID11', 'E_ID13', 'P_ID13',
                           'NE_ID17', 'NE_ID12', 'E_ID10', 'P_ID10', 'E_ID9', 'B_ID6']
    colab_test_ID = []
    test_list = pd.Series([])
    test_label = pd.Series([])
    for id in fixed_colab_test_ID:
        if all_X_list[all_X_list.index == id].empty == True:
            print(f'Test ID: {id} is missing!')
            continue
        test_list = test_list.append(all_X_list[all_X_list.index == id])
        test_label = test_label.append(all_y_list[all_y_list.index == id])
        colab_test_ID.append(id)

    return colab_test_ID, test_list, test_label

def check_transformation(video_names, exercise_type):
    # Load data
    try:
        os.makedirs(f"transformation_check_{exercise_type}/")
    except OSError as e:
        raise
    test_spatial_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((540, 500)),
        transforms.Resize((112, 112)),
        transforms.ToTensor()])

    for video in video_names:
        X = skvideo.io.vread(video, outputdict={'-r': "1"})  # (frames, height, width, channel)
        X_list = []
        for i in range(X.shape[0]):
            X_list.append(test_spatial_transform(X[i]))

        X = torch.stack(X_list, dim=0)  # [frames * channels * height * weight]
        # check input
        plt.imshow(X[0, :].permute(1, 2, 0))
        plt.savefig(f"transformation_check_{exercise_type}/" + video.split('/')[-1].split('.')[0] + ".png")
