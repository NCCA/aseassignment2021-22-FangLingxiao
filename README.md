# Music Classificaiton Project
Lingxiao Fang (ID: s5614279)

## Abstract
Music, though as an art, has the beauty of mathematics behind its patterns, which makes it highly computable. Recently, with the development of deep learning, technology with musci has made remarkable progresscc. In this project, I will implement a music classifier using the deep neural network method.

## Introduction
For this project, I convert the audio files to mel spectrograms. Conparing with traditional machine learning, which needs to acquire a series of sound features, we can use deep neural networks, such as CNN and ResNet, to analyze audio data and classify its genres. 

### GTZAN Datasets

The `GTZAN dataset` is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings. The format of the files were `.wavs`, so I was able to use the librosa library to read them into the notebook.

## Model Architecture
Deep learning doesn't require us to extract features by ourselves, but to automatically learn high-dimensional abstract data through neural networks. For the problem of speech emotion recognition, I will choose Convolution Neural Network as the deep learning model. Convolutional Neural Network has the ability of representation learning, and there is no additional feature engineering requirement for data. The figure below provides the layer-wise architecture of a basic CNN network. [1]This simple 2D CNN model consists of 3x3 convolution, ReLU non-linearity, and 2x2 max pooling. This module is going to be used for each layer of the 2D CNN.The Conv_2d class represents a generic 2D convolutional layer. [2]
![alt text](Project-Proposal/Images/cnn.png)
In addition, I use ResNet to get better result. 

## References
[1] A. Sehgal and N. Kehtarnavaz, "A Convolutional Neural Network Smartphone App for Real-Time Voice Activity Detection," in IEEE Access, vol. 6, pp. 9017-9026, 2018, doi: 10.1109/ACCESS.2018.2800728.

[2] M. Won, J. Spijkervet, and K. Choi, "Music Classification: Beyond Supervised Learning, Towards Real-world Applications" 2021, URL: https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

