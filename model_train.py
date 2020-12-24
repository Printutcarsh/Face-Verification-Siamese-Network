#Importing Libraries
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D, Lambda
import tensorflow.keras.backend as K

#Loading the pre-processed data
X = np.load("x.npy")
Y = np.load("y.npy")

#Splitting the data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=3)

# specify the shape of the inputs for our network
IMG_SHAPE = (128, 128, 3)

def build_siamese_model(inputShape, embeddingDim=100):
	# specify the inputs for the feature extractor network
    inputs = Input(inputShape)
	# define the first set - Convolution -> BatchNormalization -> Convolution -> BatchNormalization -> MaxPooling -> Dropout
    x = Conv2D(64, (5, 5), activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (5, 5), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
	# second set - Convolution -> BatchNormalization -> Convolution -> BatchNormalization -> MaxPooling -> Dropout
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    # prepare the final outputs
    x = Flatten()(x)
    outputs = Dense(embeddingDim)(x)
	# build the model
    model = Model(inputs, outputs)
	# return the model to the calling function
    return model

# configure the siamese network
#First, we create two inputs, one for each image in the pair
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
#Then builds the network architecture, which serves as featureExtractor
featureExtractor = build_siamese_model(IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
# Lastly, to construct the siamese network we need to add two more layers
#The one layer will calculate the absolute distance between the two image embeddings
#and the last layer will be our output layer which will classify the image pairs as 0 or 1
distance = Lambda( lambda tensors : K.abs( tensors[0] - tensors[1] ))( [featsA , featsB] )
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
print(model.summary())

#Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#Train the model
model.fit([X_train[:, 0], X_train[:, 1]], y_train[:], validation_data=([X_val[:, 0], X_val[:, 1]], y_val[:]), batch_size=32, epochs=30)

#Saving the model
model_json = model.to_json()
with open("model-siam.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model-siam.h5')
