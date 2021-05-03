import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from opts import parse_opts
from util import *
from plot_train import *

from datetime import datetime
from pytz import timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


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

    exercise_type = config.get('dataset', 'exercise_type')
    exercise_label_text = config.get('dataset', 'exercise_label_text')
    df_path = config.get('dataset', 'df_path')
    change_dir(df_path)
    dataset_filter = config.get('dataset', 'dataset_filter')
    df_name = exercise_type + '_' + dataset_filter + '_df'
    df = pd.read_pickle(df_name)

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

    # This is to ensure LSTM and 3d-CNN models will be tested on the same set of test data
    colab_test_ID, test_list, test_label = get_fixed_test_data(all_X_list, all_y_list)
    
    full_train_list = all_X_list[~all_X_list.index.isin(colab_test_ID)]
    full_train_label = all_y_list[~all_y_list.index.isin(colab_test_ID)]

    ########################################################################
    # Load data: [num_data x feature_dim]
    skeletal_features_path = config.get('dataset', 'skeletal_features_path')
    feat_indices = json.loads(config.get(model_name, 'feat_indices'))
    n_features = len(feat_indices)
    n_repetition = config.getint('dataset', 'n_repetition')
    input_dim = n_features * n_repetition

    train_x = test_x = np.array([]).reshape(0, input_dim)
    train_y = np.array(full_train_label.to_list())
    test_y = np.array(test_label.to_list())
    # Load data from the text file
    for video_path in full_train_list:
        txt_file_name = os.path.join(*(video_path.split('/')[-6:])).replace("/", "_").split(".")[0]

        if (txt_file_name[-1] == "_"):
            # TODO: need a better fix
            # Naming is inconsistent in KIMORE, some videos has an extra underscore. The extra underscore needs to be removed
            # i.e. 'CG_Expert_E_ID9_Es1_rgb_Blur_rgb271114_123334_'
            txt_file_name = txt_file_name[:-1]

        data = np.loadtxt(os.path.join(skeletal_features_path, txt_file_name + ".txt"), delimiter=',')
        # Handle the special case where there is 1 feature
        if len(data.shape) == 1:
            assert len(data.shape) == len(feat_indices)

        else:
            data = data[:, feat_indices]

        # Flatten the data
        data_1D = np.reshape(data, -1)
        train_x = np.vstack([train_x, data_1D])

    # Load data from the text file
    for video_path in test_list:
        txt_file_name = os.path.join(*(video_path.split('/')[-6:])).replace("/", "_").split(".")[0]

        if (txt_file_name[-1] == "_"):
            # TODO: need a better fix
            # Naming is inconsistent in KIMORE, some videos has an extra underscore. The extra underscore needs to be removed
            # i.e. 'CG_Expert_E_ID9_Es1_rgb_Blur_rgb271114_123334_'
            txt_file_name = txt_file_name[:-1]

        data = np.loadtxt(os.path.join(skeletal_features_path, txt_file_name + ".txt"), delimiter=',')
        # Handle the special case where there is 1 feature
        if len(data.shape) == 1:
            assert len(data.shape) == len(feat_indices)

        else:
            data = data[:, feat_indices]
        # Flatten the data
        data_1D = np.reshape(data, -1)
        test_x = np.vstack([test_x, data_1D])

    # Load model
    if model_name == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(300,100),
                              max_iter=1000, random_state=seed,
                              learning_rate_init=learning_rate, solver='adam')


    elif model_name == 'linearReg':
        model = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=5000, random_state=seed)

    elif model_name == 'RF':
        model = RandomForestRegressor(random_state=seed)

    elif model_name == 'SVM':
        model = SVR()

    elif 'KNN' in model_name:
        n_neighbors = config.getint(model_name, 'n_neighbors')
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

    elif model_name == 'lasso':
        model = Lasso(max_iter=1000, random_state=seed)

    model.fit(train_x, train_y)
    predict_list = model.predict(test_x)
    labels_list = test_y
    # Set up some numpy arrays to store the training/test loss/accuracy
    test_loss = mean_squared_error(test_y, predict_list)

    print("Final Test Loss: {:0.2f}".format(test_loss))

    print('Test IDs: ' + str(colab_test_ID))
    print('Test labels_list:  ' + str(list(np.around(labels_list, 2))))
    print('Test predicts_list:' + str(list(np.around(predict_list, 2))))

    # Compute Spearman correlation
    rho, pval = stats.spearmanr(predict_list, labels_list)
    print('Spearman correlation coefficient: {0:0.2f} with associated p-value: {1:0.2f}.'.format(rho, pval))


    # Evaluate the model by 5-fold cross-validation

    cv_scores_mse = cross_val_score(model, np.vstack([train_x, test_x]), np.append(train_y, test_y), cv=KFold(n_splits=5),
                                    scoring='neg_mean_squared_error')
    cv_scores_mse = -1 * cv_scores_mse
    print("5-fold Cross Validation Scores (with MSE): {} mean: {:0.2f}".format(cv_scores_mse, cv_scores_mse.mean()))

    def custom_spearmanr(x, y):
        correlation, _ = stats.spearmanr(x, y)
        return correlation
    cv_scores_spearman = cross_val_score(model, np.vstack([train_x, test_x]), np.append(train_y, test_y),cv=KFold(n_splits=5),
                                        scoring=make_scorer(custom_spearmanr, greater_is_better=True))
    print("5-fold Cross Validation Scores (with Spearman): {} mean: {:0.2f}".format(cv_scores_spearman,
                                                                                        cv_scores_spearman.mean()))

    # Create a directory with TIME_STAMP and model_name to store all outputs
    output_path = config.get('dataset', 'result_output_path')

    output_path = os.path.join(output_path, '{0}_sk_{1}_features_{2}_loss_{3:0.1f}_spear_{4:0.2f}_cvMSE_{5:0.2f}'
                                            '_cvSpear_{6:0.2f}'.format(
        TIME_STAMP, model_name, feat_indices, test_loss, rho, cv_scores_mse.mean(), cv_scores_spearman.mean()))
    try:
        os.mkdir(output_path)
        os.chdir(output_path)
    except OSError:
        print("Creation of the directory %s failed!" % output_path)

    # Save test results to txt file
    record_test_results(output_path, colab_test_ID, labels_list, predict_list, test_loss, model_name, config,
                        cv_scores_mse, cv_scores_spearman)
    # Plot test results
    plot_labels_and_outputs(labels_list, predict_list, config, model_name, colab_test_ID, test_loss)
    # Plot training loss
    # plot_training_loss(model_name, 'loss', train_loss, test_loss, config, output_path)

if __name__ == '__main__':
    main()
