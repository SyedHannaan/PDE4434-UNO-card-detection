# PDE4434-UNO-card-detection
this repository contains the dataset, brief explanation and the code to detect the colour and number of the UNO card and what machine learning is used for the same and how sensing plays role in this

----------------------------------------------------------------------------------------------------------------------------

Install the prequisite. install python install jupyter notebook install pip install cv2, numpy, tensorflow etc.

In my case I am using Python3 for Jupyter notebook using anaconda where i can preinstall libraries and python version based on the requirements

Initialise Jupter notebook and create a new python 3 program

#import all the directories and define the terms. import os import cv2 import numpy as np from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout from tensorflow.keras.preprocessing.image import ImageDataGenerator

------------------------------------------------------------------------------------------------------------------------------

The above code uses computer vision and deep learning techniques to detect and classify Uno cards in real-time using a webcam. This approach is known as object detection and recognition, which is an important problem in computer vision and has numerous applications in various fields, such as robotics, surveillance, and autonomous vehicles.

The object detection and recognition pipeline involves multiple stages, including image acquisition, preprocessing, feature extraction, object detection, and classification. The code above uses OpenCV to acquire frames from the webcam and preprocesses them by resizing and scaling the pixel values to the range [0,1]. The preprocessed frames are then passed through a pre-trained Keras model to get the predicted class label and probability.

Deep learning has shown remarkable performance in various computer vision tasks, including object detection and recognition. The Keras model used in the code above is most likely a convolutional neural network (CNN), which is a deep learning architecture widely used in image classification and object detection tasks. CNNs are designed to automatically learn features from raw image data by applying convolutional filters to the input image.

The performance of CNNs in object detection and recognition tasks heavily depends on the size and quality of the training data. In the case of the above code, the model was trained on a dataset of Uno card images, which were annotated with their corresponding class labels. The annotated images were used to train the CNN to recognize different Uno card classes and learn their distinguishing features.

The use of pre-trained models for object detection and recognition tasks has become a common practice due to the availability of large and diverse datasets, such as ImageNet, COCO, and Pascal VOC, that have been used to train CNNs. Pre-trained models can be fine-tuned on specific datasets to improve their performance on new tasks or new classes of objects.

In conclusion, the code uses computer vision and deep learning techniques to detect and classify Uno cards in real-time using a webcam. The code demonstrates the effectiveness of using pre-trained CNNs for object detection and recognition tasks, which has become a common practice in the field of computer vision.


------------------------------------------------------------------------------------------------------------------------------

This is a Python code for building a deep learning model for uno card detection. It uses the Keras API with TensorFlow backend. The dataset consists of images of uno cards for playing the game Uno. The model is trained on the dataset using data augmentation and is saved for future use. The code also contains a function for predicting the color and number of a uno card captured by a webcam.

The model architecture is a simple convolutional neural network consisting of four convolutional layers with increasing number of filters, followed by max pooling layers. The output of the convolutional layers is flattened and fed into two fully connected layers with a dropout layer in between them. The final output layer has a softmax activation function and 53 units, corresponding to the number of classes in the dataset.

The trained model is then saved to a file for future use. Finally, the code defines a function for predicting the label of a hand gesture captured by a webcam. It loads the saved model and preprocesses the input image by resizing it and scaling the pixel values to the range [0, 1]. It then makes a prediction using the model and returns the predicted label and probability.

 
 -----------------------------------------------------------------------------------------------------------------------------
 
 
 The above code implements a Convolutional Neural Network (CNN) model to recognize hand gestures of Uno cards. Here's a brief explanation of what each part of the code does:

Importing Libraries: The first few lines of the code import necessary libraries and modules such as os, cv2, numpy, keras, and ImageDataGenerator from TensorFlow.

Data Preprocessing: The next step is to preprocess the data. The ImageDataGenerator module from TensorFlow is used to create generators for training and validation data. The rescale parameter is used to scale the pixel values of the images to the range of [0, 1]. Other parameters such as shear_range, zoom_range, and horizontal_flip are used to augment the training data. The generators for the training and validation data are created using the flow_from_directory method.

Model Architecture: After preprocessing the data, a CNN model is defined using the Sequential class from Keras. The model contains several layers of Conv2D and MaxPooling2D layers, followed by a Flatten layer, two Dense layers, and a Dropout layer. The activation function used in the Conv2D layers is ReLU and the activation function in the output layer is softmax.

Compiling the Model: The compile method is used to compile the model with an optimizer of 'adam', loss function of categorical_crossentropy, and accuracy as the evaluation metric.

Training the Model: The fit method is used to train the model using the training and validation data generators created earlier. The steps_per_epoch and validation_steps are set to the number of training and validation samples divided by the batch size, respectively.

Saving the Model: After the model is trained, it is saved in a file using the save method.

Loading the Model: The saved model is then loaded into memory using the load_model function from Keras.

Image Preprocessing and Prediction: The preprocess_image function is defined to preprocess the input image. The function takes an image, converts it to grayscale, resizes it to the expected input shape, scales the pixel values to the range of [0, 1], reshapes it to have a single channel, and returns the preprocessed image. The predict_label function defines a loop that reads frames from a video camera and predicts the label of each frame. The preprocess_image function is called to preprocess each frame, and the predict method of the loaded model is called to predict the label of the frame. The predicted label and probability are then displayed on the screen.

------------------------------------------------------------------------------------------------------------------------------

DEMO VIDEO LINK : https://drive.google.com/drive/folders/1-NnoWEfIa6STcEVirvuXa7-xDeomziyo

--------------------------------------------------------------------------------------------------------------------------------

REFERENCES:

1.Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.

2.Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).

3.Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

4.Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

5.Convolutional Neural Networks:
Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, no. 7553, pp. 436-444, May 2015.

6.Transfer Learning: O. M. Parkhi, A. Vedaldi, and A. Zisserman, "Deep face recognition," in Proceedings of the British Machine Vision Conference, 2015.

7.Keras: F. Chollet et al., "Keras," 2015. [Online]. Available: https://keras.io.

8.Image Preprocessing: A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in Proceedings of the 25th International Conference on Neural Information Processing Systems, 2012, pp. 1097-1105.

9.H5PY: A. Collette, "Python and HDF5," 2013. [Online]. Available: http://docs.h5py.org/en/stable/.






