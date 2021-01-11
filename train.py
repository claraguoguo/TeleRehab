
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import generate_model
from opts import parse_opts
from load_data import KiMoReDataLoader
from plot_train import *
from util import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', DEVICE)

# Additional Info when using cuda
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def test(model, loader, criterion):
    """ Test the model on the test set.

     Args:
         model: PyTorch neural network object
         loader: PyTorch data loader for the test set
         criterion: The loss function
     Returns:
         loss: A scalar for the average loss function over the test set
     """
    total_loss = 0.0
    labels_list = []
    outputs_list = []
    predict_list = []

    total_err = 0.0
    total_labels = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        labels_list += labels.tolist()
        with torch.no_grad():
            model.eval()
            # Get model outputs
            outputs = model(inputs)
            _, predict = torch.max(outputs, 1)
            # Append outputs to list
            outputs_list += outputs.flatten().tolist()
            predict_list += predict.flatten().tolist()
            # Compute loss
            loss = criterion(outputs, labels.long())

            total_err += torch.sum(predict != labels.data)
            total_labels += len(labels)
            total_loss += loss.item()
        print('total_valid_err: {}, total size: {}'.format(total_err, total_labels))

    loss = float(total_loss) / (i + 1)
    print('Final testing labels_list:')
    print(labels_list)
    print('Final testing outputs_list:')
    print(outputs_list)
    print('Final testing predicts_list:')
    print(predict_list)
    err = float(total_err) / total_labels

    return labels_list, outputs_list, loss, err

def evaluate(model, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         model: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the average classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_labels = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
            _, predict = torch.max(outputs, 1)
            loss = criterion(outputs, labels.long())
        total_loss += loss.item()
        total_err += torch.sum(predict != labels.data)
        total_labels += len(labels)
        print('total_valid_err: {}, total size: {}'.format(total_err, total_labels))
        print(f'predict: {predict}, \nlabels: {labels.data.T}')

    loss = float(total_loss) / (i + 1)
    err = float(total_err) / total_labels

    return loss, err 

def train(epoch, model, loader, optimizer, criterion):
    total_train_loss = 0.0
    total_train_err = 0.0
    counter = 0 
    total_labels = 0
    for i, data in enumerate(loader, 0):
        model.train()
        # Get the inputs
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        _, predict = torch.max(outputs, 1)
        print(f'Epoch: {epoch}, Batch: {counter}, \npredict: {predict}, \nlabels: {labels.data.T}')
        counter += 1
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # Calculate the statistics
        total_train_loss += loss.item()
        total_train_err += torch.sum(predict != labels.data)
        total_labels += len(labels)
        print('total_train_err: {}, total size: {}, accuracy: {:0.2f}'.format(\
            total_train_err, total_labels, 1 - total_train_err/total_labels))

    loss = float(total_train_loss) / (i+1)
    err = float(total_train_err) / total_labels
    return loss, err
    
def main():

    ########################################################################
    # load args and config
    args = parse_opts()
    config = get_config(args.config)

    ########################################################################
    # Extract Frames from videos
    should_use_local_df = config.getint('dataset', 'should_use_local_df')
    if should_use_local_df:
        print('Using local df')
        df_path = config.get('dataset', 'df_path')
        change_dir(df_path)
        dataset_filter = config.get('dataset', 'dataset_filter')
        df_name = dataset_filter + '_df'
        df = pd.read_pickle(df_name)
        # TODO: Fix max_video_sec to not be hard-coded
        max_video_sec = 60
    else:
        # extract_frames_from_video(config)
        data_loader = KiMoReDataLoader(config)
        data_loader.load_data()
        df = data_loader.df
        fps = config.getint('dataset', 'fps')
        max_video_sec = data_loader.max_video_sec * fps

    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    seed = config.getint('random_state', 'seed')
    np.random.seed(seed)
    torch.manual_seed(seed)

    #######################################################################
    # Loads the configuration for the experiment from the configuration file
    model_name = args.model_name
    num_epochs = config.getint(model_name, 'epoch')
    # optimizer = config.get(model_name, 'optimizer')
    loss_fn = config.get(model_name, 'loss')
    learning_rate = config.getfloat(model_name, 'lr')
    test_size = config.getfloat('dataset', 'test_size')
    bs = config.getint(model_name, 'batch_size')

    binary_threshold = config.getint('dataset', 'binary_threshold')
    ########################################################################
    # list all data files
    all_X_list = df['video_name']                        # all video file names
    all_y_list = df['clinical TS Ex#1']                  # all video labels

    if model_name == 'binary':
        binary_labels = df
        binary_labels.loc[binary_labels['clinical TS Ex#1'] < binary_threshold, 'clinical TS Ex#1'] = 0
        binary_labels.loc[binary_labels['clinical TS Ex#1'] > binary_threshold, 'clinical TS Ex#1'] = 1
        all_y_list = binary_labels['clinical TS Ex#1']

    # transform the labels by taking Log10
    # log_all_y_list = np.log10(all_y_list)

    exercise_type = config.get('dataset', 'exercise_type')

    print('Total number of samples {} for {}'.format(all_X_list.shape[0], exercise_type))

    # train, test split
    # full_train_list --> Train_list + Validation_list    |   test_list
    full_train_list, test_list, full_train_label, test_label = train_test_split(all_X_list, all_y_list,
                                                                                test_size=test_size, random_state=seed)

    # full_train_list --> Train_list + Validation_list
    train_list, valid_list, train_label, valid_label = \
        train_test_split(full_train_list, full_train_label, test_size=0.1, random_state=seed)

    # Obtain the PyTorch data loader objects to load batches of the datasets
    full_train_loader, test_loader = get_data_loader(full_train_list, test_list, full_train_label,
                                                     test_label, model_name, max_video_sec, config)

    train_loader, valid_loader = get_data_loader(train_list, valid_list, train_label, valid_label,
                                                 model_name, max_video_sec, config)

    ########################################################################
    # Define a Convolutional Neural Network, defined in models
    model = generate_model(model_name, max_video_sec, config)
    # Load model on GPU
    model.to(DEVICE)
    ########################################################################
    # Define the Loss function, optimizer, scheduler

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)

    if loss_fn == 'l1':
        print('Loss function: nn.L1Loss()')
        criterion = nn.L1Loss()
    elif loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif loss_fn =='cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        print('Loss function: nn.MSELoss()')
        criterion = nn.MSELoss()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ########################################################################
    # Set up some numpy arrays to store the training/test loss/accuracy
    train_loss = np.zeros(num_epochs)
    train_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)

    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    print('Start training {}...'.format(model_name))

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss[epoch], train_err[epoch] = train(epoch, model, train_loader, optimizer, criterion)
        scheduler.step()
        val_loss[epoch], val_err[epoch] = evaluate(model, valid_loader, criterion)
        print("Epoch {}: Train err: {}, Train loss: {} | Validation err: {}, Validation loss: {}".format(\
            epoch + 1, train_err[epoch], train_loss[epoch], val_err[epoch], val_loss[epoch]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Train model with all training data:
    final_train_loss, final_train_err = train(epoch, model, full_train_loader, optimizer, criterion)
    print("Final Train Loss: {}, Final Train Error: {}".format(final_train_loss, final_train_err))

    # Test the final model
    labels_list, outputs_list, test_loss, test_err = test(model, test_loader, criterion)
    print("Final Test Loss: {}, Final Test Error: {}".format(test_loss, test_err))

    # Change to ouput directory and create a folder with timestamp
    output_path = config.get('dataset', 'result_output_path')
    change_dir(output_path)

    # Save the model
    should_save_model = config.getint('output', 'should_save_model')
    if should_save_model:
        torch.save(model.state_dict(), model_name)

    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    fps = config.get('dataset', 'fps')

    df = pd.DataFrame({"epoch": epochs, "train_loss": train_loss})
    df.to_csv("train_{}_loss_{}_lr{}_epoch{}_bs{}_fps{}.csv".format(model_name, loss_fn,
                                                                    learning_rate, num_epochs, bs, fps), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_loss": val_loss})
    df.to_csv("val_{}_loss_{}_lr{}_epoch{}_bs{}_fps{}.csv".format(model_name, loss_fn,
                                                                  learning_rate, num_epochs, bs, fps), index=False)

    generate_result_plots(model_name, test_loss, config)

    # Create a scatterplot of test results
    plot_labels_and_outputs(labels_list, outputs_list, config, model_name)
    plt.close()
if __name__ == '__main__':
    main()
