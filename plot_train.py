import matplotlib.pyplot as plt
import pandas as pd
import torch
from util import *


def load_csv(type, model_name, config, include_acc=False):
    """
    Given the type ('err'/'loss'), loads the appropriate CSV files to plot

    :param type: string denoting the type of files to load ('err' or 'loss')
    :param config: configuration dictionary
    :return: Numpy arrays for the train and test value
    """

    epoch = config.getint(model_name, 'epoch')
    lr = config.getfloat(model_name, 'lr')
    bs = config.getint(model_name, 'batch_size')
    loss_fn = config.get(model_name, 'loss')
    fps = config.get('dataset', 'fps')

    model_path = model_name

    train_file = 'train_{}_{}_{}_lr{}_epoch{}_bs{}_fps{}.csv'.format(model_path, type, loss_fn, lr, epoch, bs, fps)
    val_file = 'val_{}_{}_{}_lr{}_epoch{}_bs{}_fps{}.csv'.format(model_path, type, loss_fn, lr, epoch, bs, fps)
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)

    if include_acc:
        train_acc_file = "train_acc_{}_lr_{}_epoch{}_bs{}_fps{}.csv".format(model_path, lr, epoch, bs, fps)
        val_acc_file = "val_acc_{}_lr_{}_epoch{}_bs{}_fps{}.csv".format(model_path, lr, epoch, bs, fps)

        train_acc_data = pd.read_csv(train_acc_file)
        val_acc_data = pd.read_csv(val_acc_file)

        return train_data, val_data, train_acc_data, val_acc_data

    return train_data, val_data


def plot_graph(model_name, type, train_data, val_data, test_loss, config):
    """
    Plot the training loss/error curve given the data from CSV
    """
    epoch = config.getint(model_name, 'epoch')
    lr = config.getfloat(model_name, 'lr')
    bs = config.getint(model_name, 'batch_size')
    loss_fn = config.get(model_name, 'loss')
    fps = config.get('dataset', 'fps')

    plt.figure()
    plt.title("{0} over training epochs \n {1}_lr{2}_epoch{3}_bs{4}_fps{5}_test{6:.3f}".format(
        type, model_name, lr, epoch, bs, fps, test_loss))
    plt.plot(train_data["epoch"], train_data["train_{}".format(type)], label="Training")
    plt.plot(val_data["epoch"], val_data["val_{}".format(type)], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(loss_fn + type)
    plt.legend(loc='best')
    plt.savefig("{0}_{1}_{2}_lr{3}_epoch{4}_bs{5}_fps{6}_test{7:.3f}.png".format(
        model_name, type, loss_fn, lr, epoch, bs, fps, test_loss))
    plt.close()
    return

def generate_result_plots(model_name, test_loss, config, test_acc=-1):
    ########################################################################
    # Loads the configuration for the experiment from the configuration file
    # Load the CSV files according to the current config
    # train_err_data, val_err_data = load_csv('err', model_path)

    # fetch and plot accuracy data, test_acc = -1 be default
    if test_acc != -1:
        train_loss_data, val_loss_data, train_acc_data, val_acc_data = load_csv('loss', model_name, config, include_acc=True)

        # Print the final accuracy for the train/validation set from the CSV file
        print("Final training acc: {0:.2f} | Final validation acc: {1:.2f} | Final Test acc: {2:.2f}".format(
            train_acc_data["train_acc"].iloc[-1],
            val_acc_data["val_acc"].iloc[-1], test_acc))

        # Plot a train vs test accuracy graph
        plot_graph(model_name, 'acc', train_acc_data, val_acc_data, test_acc, config)

    else:
        train_loss_data, val_loss_data = load_csv('loss', model_name, config)

    # Print the final loss/error for the train/validation set from the CSV file
    print("Final training loss: {0:.5f} | Final validation loss: {1:.5f} | Final Test loss: {2:.5f}".format(train_loss_data["train_loss"].iloc[-1],
          val_loss_data["val_loss"].iloc[-1], test_loss))

    # Plot a train vs test loss graph for this hyperparameter
    plot_graph(model_name, 'loss', train_loss_data, val_loss_data, test_loss, config)
