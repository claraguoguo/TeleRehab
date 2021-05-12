# TeleRehab

This project develops an automated system for assessing physical rehabilitation exercises using RGB data.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project develops an automated system for assessing physical rehabilitation exercises using RGB data. The end-to-end deep learning approaches (i.e. C3D, 3D-ResNet) and the feature extraction based approaches (i.e. LR, MLP, KNN) were implemented. [KiMoRe](https://github.com/petteriTeikari/KiMoRe_wrapper/wiki) dataset is used for training and testing the models.


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

* python 3.7

### Installation

1. Clone the repo. Note the *colab* branch is the main branch.
   ```sh
   git clone -b colab https://github.com/claraguoguo/TeleRehab.git
   ```
2. Install common ML libraries i.e. scipy, pandas, numpy, matplotlib, seaborn, ffmpeg...

<!-- USAGE EXAMPLES -->
## Config

* config.cfg: config file for running code locally
* colab_config.cfg: config file for running code on Google Golab 
  
# Ex1
* n_repetition = 5
# Ex2
* n_repetition = 10
# Ex3
* n_repetition = 15
# Ex4
* n_repetition = 10
# Ex5
* n_repetition = 10

## Run models
* train.py: train and test deep learning models (cnn, resnet, c3d)
* train_LSTM.py: train and test LSTM model
* train_NN.py: train and test MLP and linear regression models
* train_NN_sklearn.py: train and test sklearn models 
* train_weighted_loss.py: train and test deep learning models (cnn, resnet, c3d) with weighted loss implementation
  
## sample usage: 
```
python train.py --config config.cfg --model_name c3d
```

## Related Repos
* Code to extract features from skeletal data can be found in [TeleRehab_Utilities](https://github.com/claraguoguo/TeleRehab_Utilities/SkeletalDataUtils)

* (DEPRECATED) Code used to extract skeletal joints with openpose-COCO model can be found at [demo_video_KIMORE.py](https://github.com/claraguoguo/pytorch-openpose/blob/master/demo_video_KIMORE.py). This code is developed on top of [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose).

## Useful Resources

* [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [pytorch-openpose](https://github.com/Hzzone/pytorch-openpose) - pytorch implementation of openpose including Body and Hand Pose Estimation (this version works on Apple M1 chip)


<!-- CONTACT -->
## Contact

Clara Guo  - [clara.guo@mail.utoronto.ca](clara.guo@mail.utoronto.ca)

