# DogBreedClassifier
  This project is an improvisation over the Deep learning assignment to predict Dog breed from images by Udacity. The project covers arriving at an optimal solution to the problem using __Transfer Learning__. I have currently leveraged VGG19 model to extract the bottleneck features and proceeded from there. The pretrained model and feature extraction is provided by __Keras__. 
Keras is an excellent choice to make use of Tensorflow as a black box by passing in configurable parameters like Optimizers, Loss function, Number of Epochs, Model layer structure. It lets you leverage a lot of pretrained functions and a whole lot more.
Feel free to clone the repo:octocat: and optimize the system with better network and retrain with a different pretrained model.

## Environment setup:

The libraries that are used in the project can be found in the `requirements.txt` file.

If you are using Anaconda for python environment management, do the following to install the required libraries in Mac.
```
conda create --name my_env python=3.5
source activate my_env
pip install -r requirements.txt
```

Switch the Keras to use Tensorflow as backend.
```
 KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

## Project Description

This is a modularized and scripted version of the Jupyter notebook version of the project. The script `predict.py` takes in the path of the file that contains the image to be predicted.
```
python predict.py /dogImage.jpg
```

It predicts the dog breed that is present in the image and prints it out into the console along with the confidence of the system in making that prediction. If it identifies a human face in the image, it then prints the dog breed to which the face in the image looks similar to.

Currently, the project contains a pretrained saved to model to make the prediction. It is saved in the directory `saved_models/weights.best.VGG19.hdf5`. The code to train the system and use different pretrained systems to compare the outcomes will soon be added. It uses _Haar Cascade algorithm_ to detect faces in the image. A face detector to detect facial information in the image will soon be added too.
