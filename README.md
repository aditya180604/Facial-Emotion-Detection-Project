# Facial-Emotion-Detection-Project
Project Overview
This project focuses on creating a facial emotion detection system using deep learning techniques, specifically Convolutional Neural Networks (CNN). The model is designed to classify facial expressions into seven distinct emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The project includes data preprocessing, model training, and real-time emotion recognition from video input.

Data Preparation
The dataset is split into training and testing directories, each containing images organized in subdirectories named after the respective emotions. The images are preprocessed as follows:

Loading Images: Images are loaded and resized to a consistent size of 48x48 pixels. This size is chosen to balance computational efficiency with the need for sufficient detail to detect facial expressions.
Normalization: The pixel values of the images are normalized to a range of 0 to 1.
Label Encoding: The labels (emotion categories) are converted to categorical format using one-hot encoding.
Model Architecture
The model is built using the Keras library and consists of the following layers:

Convolutional Layers: Three convolutional layers are used, each followed by a max-pooling layer. These layers help in extracting relevant features from the input images.
Flatten Layer: The output from the convolutional layers is flattened into a one-dimensional vector.
Dense Layers: Two dense (fully connected) layers are used, with the final layer outputting probabilities for each of the seven emotion categories.
The model is compiled with the Adam optimizer and categorical cross-entropy loss function, and it is evaluated using accuracy as a metric.

Training and Validation
The dataset is split into training and validation sets to monitor the model's performance during training. The model is trained for a specified number of epochs, and the training history is recorded to analyze the learning curves.

Real-Time Emotion Detection
After training, the model is used for real-time emotion detection. A Haar Cascade Classifier is employed to detect faces in video frames captured from a webcam. For each detected face, the region is extracted, resized, normalized, and passed through the CNN model to predict the emotion. The predicted emotion label is then displayed on the video feed.
