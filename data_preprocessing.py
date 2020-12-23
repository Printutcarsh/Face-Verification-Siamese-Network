#Importing Libraries
import os
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split

#Initializing output path
out_path = "C:/Users/Utcarsh/OneDrive/Desktop/atompy/trainset"
#Initializing empty list to store images and its labels
train_images = []
train_labels = []
#Assigned a unique number to each folder and this folder number will be label of each image
i=0
#Loop throuh the face images folder which was created by previous code
for dir in glob.glob("C:/Users/Utcarsh/OneDrive/Desktop/atompy/trainset/Faces/*"):
    #label = dir.split("\\")[-1]

    for img in glob.glob(os.path.join(dir,"*.jpg")):
        img = cv2.imread(img)
        train_images.append(img)
        train_labels.append(i)
    i=i+1
#Converting list to nummpy array
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print(i)

#Next our model require pair of images as X and y which will be 1 for same images
#and 0 for different images
def make_pairs(images, labels):
	#initialize two empty lists to hold the (image, image) pairs and
	#labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    #calculate the total number of classes present in the dataset
	#and then build a list of indexes for each class label that
	#provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == j)[0] for j in range(0, numClasses)]

    #loop over all images and for each image a positive and a negative image pair is made
    for idxA in range(len(images)):
		#grab the current image and label belonging to the current iteration
        currentImage = images[idxA]
        label = labels[idxA]

		#randomly pick an image that belongs to the same class label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
		#prepare a positive pair and update the images and labels lists respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])

        #grab the indices for each of the class labels not equal to
		#the current label and randomly pick an image corresponding
		#to a label not equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
		#prepare a negative pair of images and update the lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
	#return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))

#build the image pairs
(pairTrain, labelTrain) = make_pairs(train_images, train_labels)

#Lastly, saving these numpy arrays
np.save('{}/x.npy'.format(out_path), pairTrain)
np.save('{}/y.npy'.format(out_path), labelTrain)
