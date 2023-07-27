#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import warnings
warnings.filterwarnings("ignore")


# In[38]:


# Load the dataset
data = pd.read_csv('train.csv')


# In[44]:


# Extract the labels and pixel values
labels = data['label']
pixels = data.drop('label', axis=1)


# In[45]:


# Normalize the pixel values to the range [0, 1]
pixels = pixels / 255.0


# In[47]:


# Reshape the pixel values into 28x28 images
images = pixels.values.reshape(-1, 28, 28, 1)


# In[11]:


# Convert the labels to one-hot encoded vectors
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)


# In[14]:


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


# In[28]:


# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[33]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))
odel.save('trained_model.h5')


# In[34]:


# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)


# In[21]:


# Plot the training loss and accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.show()

