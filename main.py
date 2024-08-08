import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

#the block of code underneath was necessary because apple has no competent paintbrush application

"""
for i in range(2,7):
    image_path = f'digit{i}.png'
    img = Image.open(image_path)
    img = img.convert("RGB")
    pixels=img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x,y]
            if (r, g, b) != (255,255,255):
                pixels[x, y] = (0,0,0)

    output_path = f'digit{i}.png'
    img.save(output_path)
    plt.imshow(img)
    plt.show()
"""


# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the training and test data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Save the model
model.save('digits.keras')
# testing

for x in range(2, 7):
    img = cv.imread(f'digit{x}.png')[:,:,0]
    img=np.invert(np.array([img]))
    prediction = model.predict(img)
    print(np.argmax(prediction))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
