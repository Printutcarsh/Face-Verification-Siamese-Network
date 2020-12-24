#Importing Libraries
from PIL import Image
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2

#Loading HaarCascade classifier
face_cascade = cv2.CascadeClassifier("C:/Users/Utcarsh/OneDrive/Desktop/atompy/HaarCascade/haarcascade_frontalface_default.xml")

#Loading the model
json_file = open("model-siam.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model-siam.h5")

#Define function to extract faces from image
def faceExtract(test_img):
    opencvImage = cv2.imread(test_img)
    faces=face_cascade.detectMultiScale(opencvImage,scaleFactor=1.3,minNeighbors=5)
    #Then we take the image if it contains only 1 face
    if len(faces) == 1:
        for (x, y, w, h) in faces:
            #extracting the region of interest
            roi = opencvImage[y:y+h, x:x+w]
            roi = cv2.resize(roi, (128, 128))
            arr = np.array(roi)
    return arr


#Input the two images to compare and calling the faceExtract function
#Enter the full path of the image
imageA = faceExtract("path_of_image_1")
imageB = faceExtract("path_of_image_2")

#Reshaping the data
imgA = imageA
imgB = imageB
imgA = np.expand_dims(imgA, axis=-1)
imgB = np.expand_dims(imgB, axis=-1)
imgA = np.expand_dims(imgA, axis=0)
imgB = np.expand_dims(imgB, axis=0)

#Using our siamese model to make predictions on the image pair and
#indicating whether or not the images belong to the same person
preds = model.predict([imgA, imgB])
proba = preds[0][0]
if proba > 0.5:
    print("Confidence score = ", proba)
    print("It's a match")
else:
    print("Confidence score = ", proba)
    print("Not a match")

#Show the input images
cv2.imshow("imageA", imageA)
cv2.imshow("imageB", imageB)

#Press 'ESC' key to exit
interrupt = cv2.waitKey(0)
if interrupt & 0xFF == 27:
    cv2.destroyAllWindows()
