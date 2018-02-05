import cv2

class FaceDetector:
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    def faces(self, pathOfImage):
        image = cv2.imread(pathOfImage)
        grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.faceCascade.detectMultiScale(grayScaleImage)
