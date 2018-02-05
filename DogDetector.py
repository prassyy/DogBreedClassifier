import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

class DogDetector:
    def __init__(self):
        self.resnetModel = ResNet50(weights='imagenet')

    def predictDog(self, imageTensor):
        image = preprocess_input(imageTensor)
        return np.argmax(self.resnetModel.predict(image))

    def containsADog(self, imageTensor):
        prediction = self.predictDog(imageTensor)
        return ((prediction <= 268) & (prediction >= 151))
