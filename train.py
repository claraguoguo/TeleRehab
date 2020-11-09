
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
from model import generate_model
from opts import parse_opts
from util import *
import numpy as np
import time
from load_data import KiMoReDataLoader
import pandas as pd
from plot_train import *

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
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        # labels = normalize_label(labels)  # Convert labels to 0/1
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        total_loss += loss.item()
        print('Validation loss = {}'.format(loss.item()))
        total_epoch += len(labels)

    loss = float(total_loss) / (i + 1)

    return loss

def main():

    use_local_df = True
    ########################################################################
    # load args and config
    args = parse_opts()
    config = get_config(args.config)

    ########################################################################
    # Extract Frames from videos
    if use_local_df:
        print('Using local df')
        df_path = config.get('dataset', 'df_path')
        df_name = config.get('dataset', 'df_name')
        change_dir(df_path)
        df = pd.read_pickle(df_name)
        # TODO: Fix max_video_sec to not be hard-coded
        max_video_sec = 60
    else:
        # extract_frames_from_video(config)
        data_loader = KiMoReDataLoader(config)
        data_loader.load_data()
        df = data_loader.df
        max_video_sec = data_loader.max_video_sec

    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    seed = config.getint('random_state', 'seed')
    np.random.seed(seed)
    torch.manual_seed(seed)

    #######################################################################
    # Loads the configuration for the experiment from the configuration file
    if args.model_name == 'cnn':
        model_name = 'cnn'
    else:
        assert False

    num_epochs = config.getint(model_name, 'epoch')
    optimizer = config.get(model_name, 'optimizer')
    learning_rate = config.getfloat(model_name, 'lr')
    test_size = config.getfloat('dataset', 'test_size')
    bs = config.getint(model_name, 'batch_size')
    ########################################################################
    # list all data files
    all_X_list = df['video_name']                       # all video file names
    all_y_list = df['clinical TS Ex#1']                  # all video labels

    # transform the labels by taking Log10
    log_all_y_list = np.log10(all_y_list)

    exercise_type = config.get('dataset', 'exercise_type')

    print('Total number of samples {} for {}'.format(all_X_list.shape[0], exercise_type))

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, log_all_y_list, test_size=test_size, random_state=seed)
    
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, valid_loader = get_data_loader(train_list, test_list, train_label, test_label, model_name, max_video_sec, config)
    ########################################################################
    # Define a Convolutional Neural Network, defined in models
    model = generate_model(model_name, max_video_sec, config)

    ########################################################################
    # Define the Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)

    ########################################################################
    # Set up some numpy arrays to store the training/test loss/accuracy
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    print('Start training {}...'.format(model_name))

    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0

        total_epoch = 0
        counter = 0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            # labels = normalize_label(labels) # Convert labels to 0/1

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = model(inputs)

            print(f'Epoch: {epoch}, Batch: {counter}, \noutputs: {outputs.data.T}, \nlabels: {labels.data.T}')
            counter += 1

            loss = criterion(outputs, labels.float())
            print('Training loss = {}'.format(loss.item()))
            loss.backward()
            optimizer.step()

            # Calculate the statistics
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_loss[epoch] = evaluate(model, valid_loader, criterion)

        print("Epoch {}: Train loss: {} | Validation loss: {}".format(epoch + 1, train_loss[epoch], val_loss[epoch]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Change to ouput directory and create a folder with timestamp
    output_path = config.get('dataset', 'result_output_path')
    change_dir(output_path)

    # Save the model
    torch.save(model.state_dict(), model_name)

    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)

    df = pd.DataFrame({"epoch": epochs, "train_loss": train_loss})
    df.to_csv("train_loss_{}_lr{}_epoch{}_bs{}.csv".format(model_name, learning_rate, num_epochs, bs), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_loss": val_loss})
    df.to_csv("val_loss_{}_lr{}_epoch{}_bs{}.csv".format(model_name, learning_rate, num_epochs, bs), index=False)

    generate_result_plots(model_name, config)

if __name__ == '__main__':
    main()

'''
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch, i + 1, len(data_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
'''