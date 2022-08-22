#imports
import tensorflow as tf
#helper imports  
import numpy as np
import matplotlib.pyplot as plt
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
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img): # code to show gui 
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
    100*np.max(predictions_array),
    class_names[true_label]),
    color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()