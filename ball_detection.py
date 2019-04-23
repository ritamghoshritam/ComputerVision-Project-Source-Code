import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json
from google.colab.patches import cv2_imshow

with open('//content//drive//My Drive//test//model_in_json.json','r') as f:
  model_json = json.load(f)

model = model_from_json(model_json)

model.load_weights('//content//drive//My Drive//tennis.h5','r')
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

x = 32
h = 45
img = cv2.imread('//content//drive//My Drive//test//egg.jpg')
#img = cv2.imread('//content//drive//My Drive//test//test.jpg')
img = Image.fromarray(img, 'RGB')
img = img.resize((299,299))
img_d = np.array(img)
img_array = np.expand_dims(img_d, axis=0)

c = int (model.predict(img_array)[0][1])

if(c==1):
  print("Tennis Ball")
  text = "Tennis Ball"
  cv2.putText(img_d, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
  cv2_imshow(img_d)
else:
  print ("Unknown")
  text1 = "Unknown"
  cv2.putText(img_d, text1, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
  cv2_imshow(img_d)
