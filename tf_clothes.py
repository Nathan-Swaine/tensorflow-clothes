#imports
import tensorflow as tf
#helper imports  
import numpy as np
#print(tf.__version__)

#specifically import the 'fashion data' from the mnist general data set
fashion_mnist = tf.keras.datasets.fashion_mnist

#import and load the data from tensor flow
#there are 60k training images and 10k test images
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#classifacations for different images, must align with train images and train images labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#lets prepare the data
#model only works on values between 0 and 1 so lets modify the data for that 
train_images = train_images /255
test_images = test_images /255
#its important the test data is treat EXACTLY like the train data


model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # flatten the images into a single dimensional array with values between 0 and 1 this only reformats data, no learning 
  tf.keras.layers.Dense(128, activation='relu'), # creat a layer of 128 fully connected nodes, each scores the image and assiges it to one of the 10 classes / classifacations of types of clothing 
  tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', # this is how the model is updated based on its loss function  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # loss measures how accurate the model is during training, used to inform the optimiser 
  metrics=['accuracy']) # used in tandem with loss function to inform optimiser 

model.fit(train_images, train_labels, epochs=10) # fits the model to the training data

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) #lets compare to the test data set

print('\nTest accuracy:', test_acc)