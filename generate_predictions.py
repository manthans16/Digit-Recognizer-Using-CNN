#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequentialuential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[23]:


# Load the test dataset
test_data = pd.read_csv('test.csv')
test_data.head(4)


# In[6]:


# Normalize the pixel values to the range [0, 1]
test_pixels = test_data / 255.0


# In[8]:


# Reshape the pixel values into 28x28 images
test_images = test_pixels.values.reshape(-1, 28, 28, 1)


# In[11]:


# Load the trained model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.load_weights('trained_model.h5')


# In[13]:


# Generate predictions for the test dataset
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)


# In[21]:


# Create the submission file
submission = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predicted_labels})


# In[22]:


# Save the submission file
submission.to_csv('submission.csv', index=False)

