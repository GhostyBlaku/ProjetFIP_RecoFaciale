import numpy as np
import cv2
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D
from fer import FER
#%matplotlib inline

image_one = cv2.imread("photo2.jpg"); # Read the image


emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
results = emo_detector.detect_emotions(image_one)
bounding_box = results[0]["box"]
emotions = results[0]["emotions"]

# Print all captured emotions with the image
print(emotions)

#Add a box on the face
result_image = cv2.rectangle(image_one,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
              ,(255, 155, 155), 6)


#Display the face with the bounding box
#result_image = mpimg.imread('emotion.jpg')
RGB_img = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
imgplot = plt.imshow(RGB_img)
plt.show()
# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion,  emotion_score = emo_detector.top_emotion(image_one)

print(dominant_emotion, emotion_score)
