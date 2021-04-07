
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
from datetime import datetime
from pytz import timezone

# Get current (EST) time stamp
fmt = "%Y-%m-%d %H:%M:%S %Z%z"
now_time = datetime.now(timezone('US/Eastern'))
TIME_STAMP = now_time.strftime("%Y_%m_%d-%H_%M")

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

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        labels_list += labels.tolist()
        with torch.no_grad():
            model.eval()
            # Get model outputs
            outputs = model(inputs)
            # Append outputs to list
            outputs_list += outputs.flatten().tolist()
            # Compute loss
            loss = criterion(outputs, labels.float())
        total_loss += loss.item()
        print('Test loss = {}'.format(loss.item()))

    loss = float(total_loss) / (i + 1)
    print('Final testing labels_list:')
    print(labels_list)
    print('Final testing outputs_list:')
    print(outputs_list)

    return labels_list, outputs_list, loss

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

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
        total_loss += loss.item()
        print('Validation loss = {}'.format(loss.item()))

    loss = float(total_loss) / (i + 1)

    return loss

def train(epoch, model, loader, optimizer, criterion):
    total_train_loss = 0.0
    counter = 0 
    for i, data in enumerate(loader, 0):
        model.train()
        # Get the inputs
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

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
        # total_epoch += len(labels)

    return float(total_train_loss) / (i+1)
    
def main():

    ########################################################################
    # load args and config
    args = parse_opts()
    config = get_config(args.config)

    ########################################################################
    # Extract Frames from videos
    should_use_local_df = config.getint('dataset', 'should_use_local_df')
    fps = config.getint('dataset', 'fps')
    exercise_type = config.get('dataset', 'exercise_type')
    exercise_label_text = config.get('dataset', 'exercise_label_text')

    if should_use_local_df:
        print('Using local df')
        df_path = config.get('dataset', 'df_path')
        change_dir(df_path)
        dataset_filter = config.get('dataset', 'dataset_filter')
        df_name = exercise_type + '_' + dataset_filter + '_df'
        df = pd.read_pickle(df_name)
        # TODO: Fix max_video_sec to not be hard-coded
        max_video_sec = df['video_seconds'].max()
    else:
        # extract_frames_from_video(config)
        data_loader = KiMoReDataLoader(config)
        data_loader.load_data()
        df = data_loader.df
        max_video_sec = data_loader.max_video_sec

    # Maximum number of frames (will be used for zero padding)
    max_frame_num = max_video_sec * fps
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    seed = config.getint('random_state', 'seed')
    np.random.seed(seed)
    torch.manual_seed(seed)

    #######################################################################
    # Loads the configuration for the experiment from the configuration file
    model_name = args.model_name
    num_epochs = config.getint(model_name, 'epoch')
    optimizer = config.get(model_name, 'optimizer')
    loss_fn = config.get(model_name, 'loss')
    learning_rate = config.getfloat(model_name, 'lr')
    test_size = config.getfloat('dataset', 'test_size')
    bs = config.getint(model_name, 'batch_size')
    ########################################################################
    # list all data files
    all_X_list = df['video_name']                         # all video file names
    all_y_list = df[exercise_label_text]                  # all video labels

    ########################################################################
    # Change video path to skeletal video location
    should_use_skeletal_video = config.getint('dataset', 'should_use_skeletal_video')
    skeletal_video_path = config.get('dataset', 'skeletal_video_path')
    if (should_use_skeletal_video):
        f = lambda row: os.path.join(skeletal_video_path,
                                     os.path.join(*(row.video_name.split("/")[6:])).replace("/", "_").split(".")[0],
                                     'openpose.avi')
        df['skeletal_video_path'] = df.apply(f, axis=1)
        all_X_list = df['skeletal_video_path']

    # transform the labels by taking Log10
    # log_all_y_list = np.log10(all_y_list)
    ########################################################################
    # This is to ensure models will alwasy be tested on the same set of test data
    colab_test_ID = ['B_ID5', 'NE_ID6', 'P_ID6', 'B_ID1', 'S_ID9', 'NE_ID13', 'P_ID11', 'E_ID13', 'P_ID13', 'NE_ID17',
                     'NE_ID12', 'E_ID10', 'P_ID10', 'E_ID9', 'B_ID6']

    test_list = pd.Series([])
    test_label = pd.Series([])
    for id in colab_test_ID:
        assert (~all_X_list[all_X_list.index == id].empty)
        test_list = test_list.append(all_X_list[all_X_list.index == id])
        test_label = test_label.append(all_y_list[all_y_list.index == id])

    full_train_list = all_X_list[~all_X_list.index.isin(colab_test_ID)]
    full_train_label = all_y_list[~all_y_list.index.isin(colab_test_ID)]

    # Obtain the PyTorch data loader objects to load batches of the datasets
    full_train_loader, test_loader = get_data_loader(full_train_list, test_list, full_train_label,
                                                     test_label, model_name, max_frame_num, config)

    print('Total number of samples {} for {}'.format(all_X_list.shape[0], exercise_type))

    # # train, test split
    # # full_train_list --> Train_list + Validation_list    |   test_list
    # full_train_list, test_list, full_train_label, test_label = train_test_split(all_X_list, all_y_list,
    #                                                                             test_size=test_size, random_state=seed)
    #
    # # full_train_list --> Train_list + Validation_list
    # train_list, valid_list, train_label, valid_label = \
    #     train_test_split(full_train_list, full_train_label, test_size=0.1, random_state=seed)

    # # Obtain the PyTorch data loader objects to load batches of the datasets
    # full_train_loader, test_loader = get_data_loader(full_train_list, test_list, full_train_label,
    #                                                  test_label, model_name, max_frame_num, config)

    # train_loader, valid_loader = get_data_loader(train_list, valid_list, train_label, valid_label,
    #                                              model_name, max_frame_num, config)

    ########################################################################
    # Define a Convolutional Neural Network, defined in models
    model = generate_model(model_name, max_frame_num, config)
    # Load model on GPU
    model.to(DEVICE)
    ########################################################################
    # Define the Loss function, optimizer, scheduler
    if loss_fn == 'l1':
        print('Loss function: nn.L1Loss()')
        criterion = nn.L1Loss()
    if loss_fn == 'ls':
        print('Loss function: nn.MSELoss()')
        criterion = nn.MSELoss()
    else:
        print('Loss function: nn.MSELoss()')
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
    for epoch in range(num_epochs):
        train_loss[epoch] = train(epoch, model, full_train_loader, optimizer, criterion)
        scheduler.step()

        # val_loss[epoch] = evaluate(model, valid_loader, criterion)
        print("Epoch {}: Train loss: {}".format(epoch + 1, train_loss[epoch]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    #
    # # Train model with all training data:
    # final_train_loss = train(epoch, model, full_train_loader, optimizer, criterion)
    # print("Final Train loss: {}".format(final_train_loss))

    # Test the final model
    labels_list, outputs_list, test_loss = test(model, test_loader, criterion)
    print("Final Test loss: {}".format(test_loss))


    print('Test IDs: ' + str(colab_test_ID))
    print('Test labels_list:  ' + str(list(np.around(np.array(labels_list), 2))))
    print('Test predicts_list:' + str(list(np.around(np.array(outputs_list), 2))))

    # Compute Spearman correlation
    rho, pval = stats.spearmanr(outputs_list, labels_list)
    print('Spearman correlation coefficient: {0:0.2f} with associated p-value: {1:0.2f}.'.format(rho, pval))

    # Change to output directory and create a folder with timestamp
    output_path = config.get('dataset', 'result_output_path')

    # Create a directory with TIME_STAMP and model_name to store all outputs
    output_path = os.path.join(output_path, '{0}_{1}_loss_{2:0.1f}_spearman_{3:0.2f}'.format(
        TIME_STAMP, model_name, test_loss, rho))
    if should_use_skeletal_video:
        output_path += '_Skeletal_video'
    try:
        os.mkdir(output_path)
        os.chdir(output_path)
    except OSError:
        print("Creation of the directory %s failed!" % output_path)


    # Save the model
    should_save_model = config.getint('output', 'should_save_model')
    if should_save_model:
        torch.save(model.state_dict(), model_name)

    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)

    df = pd.DataFrame({"epoch": epochs, "train_loss": train_loss})
    df.to_csv("train_{}_loss_{}_lr{}_epoch{}_bs{}_fps{}.csv".format(model_name, loss_fn,
                                                                    learning_rate, num_epochs, bs, fps), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_loss": val_loss})
    df.to_csv("val_{}_loss_{}_lr{}_epoch{}_bs{}_fps{}.csv".format(model_name, loss_fn,
                                                                  learning_rate, num_epochs, bs, fps), index=False)

    generate_result_plots(model_name, test_loss, config)

    # Create a scatterplot of test results
    plot_labels_and_outputs(labels_list, outputs_list, config, model_name, colab_test_ID, test_loss)

    # Save test results to txt file
    record_test_results(os.getcwd(), colab_test_ID, labels_list, outputs_list, test_loss, model_name, config)

if __name__ == '__main__':
    main()
