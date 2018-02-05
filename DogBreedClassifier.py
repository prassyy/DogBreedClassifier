import numpy as np
import pickle

from glob import glob
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Activation
from keras.models import Sequential
from DogDetector import DogDetector
from FaceDetector import FaceDetector

class DogBreedClassifier:
    def __init__(self, pathOfImage):
        self.humanFaceDetector = FaceDetector()
        self.dogDetector = DogDetector()

        self.facesInImage = self.humanFaceDetector.faces(pathOfImage)
        targetImage = image.load_img(pathOfImage, target_size=(224, 224))
        imageArray = image.img_to_array(targetImage)
        self.imageTensor = np.expand_dims(imageArray, axis=0)
        self.imageContainsADog = self.dogDetector.containsADog(self.imageTensor)
        with open("dogBreedNames.txt", "rb") as fp:
            self.dogBreedNames = pickle.load(fp)

    def predictBreedInImage(self):
        if self.imageContainsADog:
            prediction = self.predictBreed(self.imageTensor)
            if prediction[0][1] > 0.7:
                return ("Hi! the dog in the image is {}.. Prediction Confidence {:.2f}".format(prediction[0][0], prediction[0][1] * 100.))
            else:
                return ("Hi! the dog in the picture looks like {:.2f}% {} and {:.2f}% {}".format(prediction[0][1] * 100., prediction[0][0], prediction[1][1] * 100., prediction[1][0]))
        elif self.facesInImage.count > 0:
            prediction = self.predictBreed(self.imageTensor)
            return ("Hi Human! You look like {:.2f}% {} and {:.2f}% {}".format(prediction[0][1] * 100., prediction[0][0], prediction[1][1] * 100., prediction[1][0]))
        else:
            return ('Sorry, we currently offer our service only to dogs and humans who look like dogs. If you are a self obsessed cat or something, stay tuned for our updates!!')

    def predictBreed(self, imageTensor):
        bottleneck_feature = VGG19(weights='imagenet', include_top=False).predict(preprocess_input(imageTensor))

        VGG19_model = Sequential()
        VGG19_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
        VGG19_model.add(Dropout(0.25))
        VGG19_model.add(BatchNormalization())
        VGG19_model.add(Activation('elu'))
        VGG19_model.add(Dense(133, activation='softmax'))
        VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')

        predicted_vector = VGG19_model.predict(bottleneck_feature)
        predictionLikelihoodOrder = np.argsort(-predicted_vector)

        prediction = []
        for predictionIndex in predictionLikelihoodOrder:
            prediction.append((self.dogBreedNames[predictionIndex[0]], predicted_vector.item(predictionIndex[0])))
        return prediction
