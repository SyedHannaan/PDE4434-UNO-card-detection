# PDE4434-UNO-card-detection
this repository contains the dataset, brief explanation and the code to detect the colour and number of the UNO card and what machine learning is used for the same and how sensing plays role in this

This is a Python code for building a deep learning model for uno card detection. It uses the Keras API with TensorFlow backend. The dataset consists of images of uno cards for playing the game Uno. The model is trained on the dataset using data augmentation and is saved for future use. The code also contains a function for predicting the color and number of a uno card captured by a webcam.

The model architecture is a simple convolutional neural network consisting of four convolutional layers with increasing number of filters, followed by max pooling layers. The output of the convolutional layers is flattened and fed into two fully connected layers with a dropout layer in between them. The final output layer has a softmax activation function and 53 units, corresponding to the number of classes in the dataset.

The trained model is then saved to a file for future use. Finally, the code defines a function for predicting the label of a hand gesture captured by a webcam. It loads the saved model and preprocesses the input image by resizing it and scaling the pixel values to the range [0, 1]. It then makes a prediction using the model and returns the predicted label and probability.

 
 
 
 
 The above code implements a Convolutional Neural Network (CNN) model to recognize hand gestures of Uno cards. Here's a brief explanation of what each part of the code does:

Importing Libraries: The first few lines of the code import necessary libraries and modules such as os, cv2, numpy, keras, and ImageDataGenerator from TensorFlow.

Data Preprocessing: The next step is to preprocess the data. The ImageDataGenerator module from TensorFlow is used to create generators for training and validation data. The rescale parameter is used to scale the pixel values of the images to the range of [0, 1]. Other parameters such as shear_range, zoom_range, and horizontal_flip are used to augment the training data. The generators for the training and validation data are created using the flow_from_directory method.

Model Architecture: After preprocessing the data, a CNN model is defined using the Sequential class from Keras. The model contains several layers of Conv2D and MaxPooling2D layers, followed by a Flatten layer, two Dense layers, and a Dropout layer. The activation function used in the Conv2D layers is ReLU and the activation function in the output layer is softmax.

Compiling the Model: The compile method is used to compile the model with an optimizer of 'adam', loss function of categorical_crossentropy, and accuracy as the evaluation metric.

Training the Model: The fit method is used to train the model using the training and validation data generators created earlier. The steps_per_epoch and validation_steps are set to the number of training and validation samples divided by the batch size, respectively.

Saving the Model: After the model is trained, it is saved in a file using the save method.

Loading the Model: The saved model is then loaded into memory using the load_model function from Keras.

Image Preprocessing and Prediction: The preprocess_image function is defined to preprocess the input image. The function takes an image, converts it to grayscale, resizes it to the expected input shape, scales the pixel values to the range of [0, 1], reshapes it to have a single channel, and returns the preprocessed image. The predict_label function defines a loop that reads frames from a video camera and predicts the label of each frame. The preprocess_image function is called to preprocess each frame, and the predict method of the loaded model is called to predict the label of the frame. The predicted label and probability are then displayed on the screen.



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






