#Importing Libraries
from PIL import Image
import os
import sys
import numpy as np
import cv2
import imghdr
from pathlib import Path

#Input the folder name of the dataset
directory = sys.argv[1]
#Path where the extracted faces from images will be saved
OUTPUT = os.path.join(directory, "Faces")
print(OUTPUT)

#Loading HaarCascade classifier
face_cascade = cv2.CascadeClassifier("HaarCascade/haarcascade_frontalface_default.xml")

#Defining a function to extract the faces and save them
def main(file, folder, count):
    #determining the extension of the image
    type_img = imghdr.what(file)
    ext = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
    if type_img in ext:
        try:
            print(file)
            image = Image.open(file)
            opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            #Given an image and it returns rectangle co-ordinates of faces detected
            faces = face_cascade.detectMultiScale(opencvImage, 1.3, 5)
            #Then we take only those images which contain only 1 face
            if len(faces) == 1:
                for (x, y, w, h) in faces:
                    #extracting the region of interest
                    roi = opencvImage[y:y+h, x:x+w]
                    roi = cv2.resize(roi, (128, 128))
                    arr = np.array(roi)
                    #Then we will save the image in the desired folder
                    #But we will save the image in the same folder name which was given in the dataset
                    #and for that we first gave the path where the folder will be created
                    #and then its required folder name
                    FOLDER = os.path.join(OUTPUT, os.path.basename(folder))
                    #If the folder does not exist, we will first create the folder
                    if not os.path.exists(FOLDER):
                            os.makedirs(FOLDER)
                    #Next we assign the name to the image of face which is extracted
                    #and the same will be its folder name(as it's the name of person)
                    #with the number joining it which count the number of each person's image
                    output_file_name = os.path.join(FOLDER, os.path.basename(folder)+"_"+str(count)+".jpg")
                    print(output_file_name)
                    #Saving the image
                    cv2.imwrite(output_file_name, arr)
                    continue
        except:
            print("Something went wrong")

#Defining function to loop over all the images in the dataset folder
def recur(folder_path):
    #Provide path of the folder
    p = Path(folder_path)
    #Store all the files in that path
    dirs = p.glob("*")
    #Track a count of the images of each unique folder
    i=0
    for folder in dirs:
        #print(folder)
        #If its a folder the function will be called again
        if folder.is_dir():
            recur(folder)
        #Else if it is an image then the main function is called
        else:
            i+=1
            main(folder, folder_path, i)

#Calling the recur function
recur(directory)
