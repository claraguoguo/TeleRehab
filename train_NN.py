import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from opts import parse_opts
from util import *
from load_data import *
from plot_train import *

from datetime import datetime
from pytz import timezone

# Get current (EST) time stamp
fmt = "%Y-%m-%d %H:%M:%S %Z%z"
now_time = datetime.now(timezone('US/Eastern'))
TIME_STAMP = now_time.strftime("%Y_%m_%d-%H_%M")

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test(model, test_loader, criterion):
    total_loss = 0.0
    labels_list = []
    predict_list = []
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        labels_list += labels.flatten().tolist()
        with torch.no_grad():
            model.eval()
            # Get model outputs
            predictions = model(inputs.float())

            predict_list += predictions.flatten().tolist()

            # Compute loss
            loss = criterion(input=predictions.squeeze(), target=labels.float())
            total_loss += loss.item()

    test_loss = float(total_loss) / (i + 1)
    return test_loss, predict_list, labels_list


def train(epoch, model, loader, optimizer, criterion):
    total_train_loss = 0.0
    counter = 0
    total_labels = 0
    for i, data in enumerate(loader, 0):
        model.train()
        # Get the inputs
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        predictions = model(inputs.float())

        loss = criterion(input=predictions.squeeze(), target=labels.float())

        print(f'Epoch: {epoch}, Batch: {counter},'
              f' \nlabels: {labels.data.T.squeeze()} \noutputs: {predictions.squeeze()},'
              f' loss = {loss.item()}')
        counter += 1

        loss.backward()
        optimizer.step()

        # Calculate loss statistics
        total_train_loss += loss.item()
        total_labels += len(labels)

    loss = float(total_train_loss) / (i + 1)
    return loss

def main():
    # Load args and config
    args = parse_opts()
    config = get_config(args.config)

    model_name = args.model_name
    num_epochs = config.getint(model_name, 'epoch')
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

    learning_rate = config.getfloat(model_name, 'lr')
    num_epochs = config.getint(model_name, 'epoch')

    # Fixed PyTorch random seed for reproducible result
    seed = config.getint('random_state', 'seed')
    np.random.seed(seed)
    torch.manual_seed(seed)

    ########################################################################
    # Load data from dataframe
    all_X_list = df['video_name']                         # all video file names
    all_y_list = df[exercise_label_text]                  # all video labels

    # This is to ensure different models will be tested on the same set of test data
    colab_test_ID, test_list, test_label = get_fixed_test_data(all_X_list, all_y_list)

    full_train_list = all_X_list[~all_X_list.index.isin(colab_test_ID)]
    full_train_label = all_y_list[~all_y_list.index.isin(colab_test_ID)]

    # Obtain the PyTorch data loader objects to load batches of the datasets
    full_train_loader, test_loader = get_nn_data_loader(full_train_list, test_list, full_train_label,
                                                     test_label, model_name, config)

    # Load model
    model = generate_model(model_name, 0, config)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Set up some numpy arrays to store the training/test loss/accuracy
    train_loss = np.zeros(num_epochs)

    print('Start training {}...'.format(model_name))
    for epoch in range(num_epochs):
        train_loss[epoch] = train(epoch, model, full_train_loader, optimizer, criterion)
        scheduler.step()
        print("Epoch {}: Train loss: {}".format( \
            epoch + 1, train_loss[epoch]))

    print('Start testing {}...'.format(model_name))
    test_loss, predict_list, labels_list = test(model, test_loader, criterion)
    print("Final Test Loss: {:0.2f}".format(test_loss))

    print('Test IDs: ' + str(colab_test_ID))
    print('Test labels_list:  ' + str(list(np.around(np.array(labels_list), 2))))
    print('Test predicts_list:' + str(list(np.around(np.array(predict_list), 2))))

    # Compute Spearman correlation
    rho, pval = stats.spearmanr(predict_list, labels_list)
    print('Spearman correlation coefficient: {0:0.2f} with associated p-value: {1:0.2f}.'.format(rho, pval))

    # Create a directory with TIME_STAMP and model_name to store all outputs
    output_path = config.get('dataset', 'result_output_path')

    feat_indices = json.loads(config.get(model_name, 'feat_indices'))
    num_features = len(feat_indices)

    output_path = os.path.join(output_path, '{0}_{5}_{1}_features_{2}_loss_{3:0.1f}_spearman_{4:0.2f}'.format(
        TIME_STAMP, model_name, feat_indices, test_loss, rho, exercise_type))
    try:
        os.mkdir(output_path)
        os.chdir(output_path)
    except OSError:
        print("Creation of the directory %s failed!" % output_path)

    # Save test results to txt file
    record_test_results(output_path, colab_test_ID, labels_list, predict_list, test_loss, model_name, config)
    # Plot test results
    plot_labels_and_outputs(labels_list, predict_list, config, model_name, colab_test_ID, test_loss)
    # Plot training loss
    # plot_training_loss(model_name, 'loss', train_loss, test_loss, config, output_path)

if __name__ == '__main__':
    main()