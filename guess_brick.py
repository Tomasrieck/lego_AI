import os
import random
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import json

tf.config.run_functions_eagerly(True)
loaded_model = load_model("parameters.h5")
file = open("itemNo_dict.json")
itemNo_dict = json.load(file)
img_size = 400


def get_pixels(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32) / 255.0

    return img


filenames = os.listdir("bricks/dataset")
rndm = random.randrange(len(filenames))
label = filenames[rndm].split(" ")[0]
img = get_pixels(f"bricks/dataset/{filenames[rndm]}")


prediction = loaded_model.predict(img)
predicted_label = np.argmax(prediction, axis=1)

plt.imshow(img[0])
plt.show()
print("Actual Lego Item No:", label)
print("Predicted Lego item No:", list(itemNo_dict.keys())[predicted_label[0]])
