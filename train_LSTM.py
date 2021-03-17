import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from opts import parse_opts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from model import generate_model
from opts import parse_opts
from util import *
from plot_train import *
from load_data import KiMoReDataLoader

from datetime import datetime
from pytz import timezone

# Get current (EST) time stamp
fmt = "%Y-%m-%d %H:%M:%S %Z%z"
now_time = datetime.now(timezone('US/Eastern'))
TIME_STAMP = now_time.strftime("%Y_%m_%d-%H_%M")

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
N_CLASSES = 2
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

    total_corr = 0.0
    total_labels = 0
    accuracy = 0

    outputs_tensor = torch.empty((0,2)).to(DEVICE)
    binarized_labels = np.empty((0, 3), int)

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        labels_list += labels.tolist()
        with torch.no_grad():
            model.eval()
            # Get model outputs
            outputs = model(inputs)
            _, predict = torch.max(outputs, 1)

            # Append outputs with label = 1 to list
            # outputs_list += outputs[:, 1].flatten().tolist()

            # The following 2 variables are needed when calculated ROC-AUC
            if DEVICE.type == 'cpu':
                binaried_y = label_binarize(labels, classes=[0, 1, 2])
            else:
                binaried_y = label_binarize(labels.view(-1).cpu(), classes=[0, 1, 2])
            binarized_labels = np.concatenate((binarized_labels, binaried_y), 0)
            outputs_tensor = torch.cat((outputs_tensor, outputs), 0)

            predict_list += predict.flatten().tolist()

            # Compute loss
            loss = criterion(outputs, labels)

            total_corr += torch.sum(predict == labels.data)
            total_labels += len(labels)
            total_loss += loss.item()
            accuracy = total_corr / total_labels
        print('total_corr: {}, total size: {}, accuracy: {:0.2f}'.format(total_corr, total_labels, accuracy))

    loss = float(total_loss) / (i + 1)
    print('testing labels_list:')
    print(labels_list)
    # print('testing outputs_list:')
    # print(outputs_list)
    print('testing predicts_list:')
    print(predict_list)

    return binarized_labels, outputs_tensor, labels_list, predict_list, loss, accuracy

def train(epoch, model, loader, optimizer, criterion):
    total_train_loss = 0.0
    total_train_corr = 0.0
    counter = 0
    total_labels = 0
    accuracy = 0
    for i, data in enumerate(loader, 0):
        model.train()
        # Get the inputs
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        _, predict = torch.max(outputs, 1)
        print(f'Epoch: {epoch}, Batch: {counter}, \nlabels: {labels.data.T} \noutputs: {predict}')
        counter += 1

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Calculate loss & accuracy statistics
        total_train_loss += loss.item()
        total_train_corr += torch.sum(predict == labels.data)
        total_labels += len(labels)
        accuracy = total_train_corr / total_labels
        print('total_train_corr: {}, total size: {}, accuracy: {:0.2f}'.format( \
            total_train_corr, total_labels, accuracy))

    loss = float(total_train_loss) / (i + 1)
    return loss, accuracy


def main():
    ########################################################################
    # Load args and config
    args = parse_opts()
    config = get_config(args.config)

    #######################################################################
    # Loads the configuration for the experiment from the configuration file
    model_name = args.model_name
    num_epochs = config.getint(model_name, 'epoch')
    loss_fn = config.get(model_name, 'loss')
    learning_rate = config.getfloat(model_name, 'lr')
    test_size = config.getfloat('dataset', 'test_size')
    bs = config.getint(model_name, 'batch_size')
    binary_threshold = config.getint('dataset', 'binary_threshold')
    should_use_local_df = config.getint('dataset', 'should_use_local_df')

    fps = config.getint('dataset', 'fps')
    exercise_type = config.get('dataset', 'exercise_type')
    exercise_label_text = config.get('dataset', 'exercise_label_text')

    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    seed = config.getint('random_state', 'seed')
    np.random.seed(seed)
    torch.manual_seed(seed)

    ########################################################################
    # Load dataframe that has video names and clinical scores
    if should_use_local_df:
        df_path = config.get('dataset', 'df_path')
        change_dir(df_path)
        dataset_filter = config.get('dataset', 'dataset_filter')
        df_name = exercise_type + '_' + dataset_filter + '_df'
        df = pd.read_pickle(df_name)
        print('Using local df: '+ df_name)
        max_video_sec = 60
    else:
        # extract_frames_from_video(config)
        data_loader = KiMoReDataLoader(config)
        data_loader.load_data()
        df = data_loader.df
        max_video_sec = data_loader.max_video_sec
        print('max_video_sec = ' + str(max_video_sec))

    # TODO: FIX n_steps
    n_steps = max_video_sec * fps
    ########################################################################
    # Load data from dataframe
    all_X_list = df['video_name']                         # all video file names
    all_y_list = df[exercise_label_text]                  # all video labels

    # Encode the clinical scores to binary labels {0, 1} based on binary_threshold
    binary_labels = df
    binary_labels.loc[binary_labels[exercise_label_text] <= binary_threshold, exercise_label_text] = 0
    binary_labels.loc[binary_labels[exercise_label_text] > binary_threshold, exercise_label_text] = 1
    all_y_list = binary_labels[exercise_label_text].astype(int)

    ########################################################################
    # train, test split
    # full_train_list --> Train_list + Validation_list    |   test_list
    full_train_list, test_list, full_train_label, test_label = train_test_split(all_X_list, all_y_list,
                                                                                test_size=test_size, random_state=seed)

    # full_train_list --> Train_list + Validation_list
    train_list, valid_list, train_label, valid_label = \
        train_test_split(full_train_list, full_train_label, test_size=0.1, random_state=seed)

    # Obtain the PyTorch data loader objects to load batches of the datasets
    full_train_loader, test_loader = get_lstm_data_loader(full_train_list, test_list, full_train_label,
                                                     test_label, model_name, n_steps, config)

    train_loader, valid_loader = get_lstm_data_loader(train_list, valid_list, train_label, valid_label,
                                                 model_name, n_steps, config)

    ########################################################################
    # Load model
    model = generate_model(model_name, n_steps, config)
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
    elif loss_fn == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        print('Loss function: nn.MSELoss()')
        criterion = nn.MSELoss()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/accuracy
    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    print('Start training {}...'.format(model_name))

    for epoch in range(num_epochs):
        train_loss[epoch], train_acc[epoch] = train(epoch, model, train_loader, optimizer, criterion)
        scheduler.step()
        _, _, _, _, val_loss[epoch], val_acc[epoch] = test(model, valid_loader, criterion)
        print("Epoch {}: Train acc: {}, Train loss: {} | Valid acc: {}, Valid loss: {}".format( \
            epoch + 1, train_acc[epoch], train_loss[epoch], val_acc[epoch], val_loss[epoch]))

    print('Finished Training')


    # Train model with all training data:
    print('Training model with all data...')
    final_train_loss, final_train_acc = train(epoch, model, full_train_loader, optimizer, criterion)
    print("Final Train Loss: {:0.2f}, Final Train Accuracy: {:0.2f}".format(final_train_loss, final_train_acc))

    # Test the final model
    binarized_labels, outputs_tensor, labels_list, predict_list, test_loss, test_acc = test(model, test_loader,
                                                                                            criterion)
    print("Final Test Loss: {:0.2f}, Final Test Accuracy: {:0.2f}".format(test_loss, test_acc))

    # Change to output directory and create a folder with timestamp
    output_path = config.get('dataset', 'result_output_path')

    # Create a directory with TIME_STAMP and model_name to store all outputs
    output_path = os.path.join(output_path, TIME_STAMP + "_" + model_name)

    try:
        os.mkdir(output_path)
        os.chdir(output_path)
    except OSError:
        print("Creation of the directory %s failed!" % output_path)

    # show metrics evaluated on binary classifier
    pos_prob = np.asarray(outputs_tensor[:, 1].flatten().tolist())  # predicted possibilities for Class 1
    show_binary_classifier_metrics(labels_list, predict_list, pos_prob, model_name, config)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(N_CLASSES):
        i_labels = binarized_labels[:, i]
        i_outputs = np.asarray(outputs_tensor[:, i].flatten().tolist())

        fpr[i], tpr[i], _ = metrics.roc_curve(i_labels, i_outputs, pos_label=i)
        auc_2 = metrics.auc(fpr[i], tpr[i])

        # calculate AUC
        roc_auc[i] = metrics.roc_auc_score(i_labels, i_outputs)
        print('Class: {} |  metrics.roc_curve: {:0.3f} | metrics.auc: {:0.3f}'.format(i, roc_auc[i], auc_2))

    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(binarized_labels[:,:-1], outputs_tensor.flatten().tolist())
    # roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plot_ROC(fpr, tpr, roc_auc, N_CLASSES, model_name, config)

    # Save the model
    should_save_model = config.getint('output', 'should_save_model')
    if should_save_model:
        saved_model_name = f'{model_name}_epoch{num_epochs}.pk'
        torch.save(model.state_dict(), saved_model_name)

    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    fps = config.get('dataset', 'fps')

    df = pd.DataFrame({"epoch": epochs, "train_loss": train_loss})
    df.to_csv("train_{}_loss_{}_lr{}_epoch{}_bs{}_fps{}.csv".format(model_name, loss_fn,
                                                                    learning_rate, num_epochs, bs, fps), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_loss": val_loss})
    df.to_csv("val_{}_loss_{}_lr{}_epoch{}_bs{}_fps{}.csv".format(model_name, loss_fn,
                                                                  learning_rate, num_epochs, bs, fps), index=False)
    df = pd.DataFrame({"epoch": epochs, "train_acc": train_acc})
    df.to_csv("train_acc_{}_lr_{}_epoch{}_bs{}_fps{}.csv".format(model_name, learning_rate, num_epochs, bs, fps),
              index=False)

    df = pd.DataFrame({"epoch": epochs, "val_acc": val_acc})
    df.to_csv("val_acc_{}_lr_{}_epoch{}_bs{}_fps{}.csv".format(model_name, learning_rate, num_epochs, bs, fps),
              index=False)

    generate_result_plots(model_name, test_loss, config, test_acc)

    plt.close()

if __name__ == '__main__':
    main()
