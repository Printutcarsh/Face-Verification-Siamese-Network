# Face-Verification-Siamese-Network
A Face Verification Siamese Network implemented using Keras.
# Installation
• Make sure that you have Python 3.6 in your system.<br>
• Now we need to install the essentials using the command :
```
pip install opencv-contrib-python
pip install pathlib
pip install tensorflow
pip install Pillow
```
# Step By Step Implementation Guide
<ul>
  <li> The first step is to download all the files in the same folder(except Semantic Segmentation.docx). </li>
  <li> Afterthat as we are focused on faces so we have to first extract faces from the images and store them in a folder.<br>
    For that run face_extract.py in your command prompt and also write the name of the dataset folder in the command prompt like this:-<br>
    
```python face_extract.py trainset```

<br> This will output a folder named Faces which contain folders each containing faces of that person.</li>
  <li> Then we have to process the data and have to make image pairs and its corresponding label and save them.<br>
    For this run:-
  
``` python data_preprocessing.py```

  This will output x.npy and y.npy which will be used for training the model. To download x.npy, <a href = "https://drive.google.com/file/d/1RRwgA3ykN7uosnKe_qlExjosn9pB15aD/view?usp=sharing">Click here</a></li>
  <li> Next, the model is custom built using Keras and it is trained using the preprocessed data.<br>
    For this run:-
  
```python model_train.py```

  This will output the model and its weight naming model-siam.json and model-siam.h5.</li>
  <li> Finally, we will use this trained model and will apply on two images to compare whether they are of same person or not.
    For this run:-
  
```python predict.py```
</li>
</ul>

## Direct Prediction
If you want to just predict on two images, just skip all the steps and download the model-siam.json, model-siam.h5 and predict.py in the same folder. Then enter the path of the two images in the predict.py code and then run the code. The output will be printed telling whether it is a match or not with the confidence score.
## How Semantic Segmentation or FCNs are used in Autonomous Vehicles
Please refer my write up which I have uploaded as Semantic Segmentation.docx for how Semantic Segmentation or FCNs are used in Autonomous Vehicles.
