import os
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

filenames = os.listdir("bricks/dataset")
random.shuffle(filenames)
img_size = 400


def get_pixels(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0

    return img


images = []
labels = []
for fn in filenames:
    images.append(get_pixels(f"bricks/dataset/{fn}"))
    labels.append(int(fn.split(" ")[0]))

unique_labels = set(labels)
labels_dict = {}
for idx, i in enumerate(unique_labels):
    labels_dict[i] = idx

for idx, l in enumerate(labels):
    labels[idx] = labels_dict[l]

train_size = int(len(labels) * 0.8)
val_size = train_size + int(len(labels) * 0.17)

train_images = tf.convert_to_tensor(images[:train_size])
train_labels = tf.convert_to_tensor(labels[:train_size])

val_images = tf.convert_to_tensor(images[train_size:val_size])
val_labels = tf.convert_to_tensor(labels[train_size:val_size])

test_images = tf.convert_to_tensor(images[val_size:])
test_labels = tf.convert_to_tensor(labels[val_size:])

model = Sequential(
    [
        Input(shape=(img_size, img_size, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dense(50, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(
    train_images, train_labels, epochs=10, validation_data=(val_images, val_labels)
)

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

accuracy = accuracy_score(test_labels, predicted_labels)
rndm = random.randrange(len(test_labels))
plt.imshow(test_images[rndm])
plt.show()
print(f"Overall accuracy: {round(accuracy*100, 2)}%")
print(
    "Actual:",
    int(test_labels[rndm]),
    f"(Lego item No: {list(labels_dict.keys())[int(test_labels[rndm])]})",
)
print(
    "Prediction:",
    predicted_labels[rndm],
    f"(Lego item No: {list(labels_dict.keys())[predicted_labels[rndm]]})",
)

model.save("parameters.h5")
