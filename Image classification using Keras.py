
# coding: utf-8

# In[2]:


from keras.layers import Dense,activations,MaxPooling2D
from keras.models import Sequential

from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Data load
(train_imgs, train_labels),(test_imgs, test_labels)=mnist.load_data()


# In[4]:


# Data Checkout
print ("The number of training examples is: ", train_imgs.shape[0])
print ("The number of test examples is: ", test_imgs.shape[0])
print ("The size of every img is: ", train_imgs.shape[1:])
num_classes=len(np.unique(train_labels))
print ("The number of classes is: ", num_classes)


# In[5]:


# printing out some samples
figure,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))

ax1.imshow(train_imgs[7,:,:],cmap='gray')
ax1.set_title("Ground truth is: {}".format(train_labels[7]))

ax2.imshow(train_imgs[20,:,:],cmap='gray')
ax2.set_title("Ground truth is: {}".format(train_labels[20]))


# In[6]:


# Reshaping the input to be a vector instead of an array
dim_array=np.prod(train_imgs.shape[1:])
train_data=train_imgs.reshape(train_imgs.shape[0],dim_array)
test_data=test_imgs.reshape(test_imgs.shape[0],dim_array)


# In[7]:


# Convert images to png (scale from 0 to 1)
train_data=train_data.astype('float32')
test_data=test_data.astype('float32')

train_data/=255
test_data/=255


# In[8]:


#Converting labels to one-hot encoding form
from keras.utils.np_utils import to_categorical

train_labels_one_hot=to_categorical(train_labels)
test_labels_one_hot=to_categorical(test_labels)


# In[9]:


print("Before encoding",train_labels[5])
print("After encoding",train_labels_one_hot[5])


# In[10]:


#Create a network

model=Sequential()
model.add(Dense(512,activation='sigmoid',input_shape=(dim_array,)))
model.add(Dense(512,activation='sigmoid'))
model.add(Dense(num_classes,activation='softmax'))


# In[11]:


# Configure a network
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[43]:


# train a model 
history = model.fit(train_data, train_labels_one_hot, batch_size=256, nb_epoch=20, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))


# In[44]:


# test 
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


# In[49]:


plt.figure(figsize=[20,10])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss','Validation loss'],fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('loss',fontsize=16)
plt.title('Loss curves',fontsize=16)
plt.grid()


# In[50]:


plt.figure(figsize=[20,10])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy','Validation Accuracy'],fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy curves',fontsize=16)
plt.grid()


# In[15]:


# Check how dropout will affect validation accuracy

from keras.layers import Dropout

model=Sequential()
model.add(Dense(512,activation='sigmoid',input_shape=[dim_array,]))
model.add(Dropout(0.5))
model.add(Dense(512,activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))


# In[16]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[17]:


history = model.fit(train_data, train_labels_one_hot, batch_size=256, nb_epoch=20, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))


# In[18]:


plt.figure(figsize=[20,10])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss','Validation loss'],fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('loss',fontsize=16)
plt.title('Loss curves using Dropout regularization',fontsize=16)
plt.grid()


# In[19]:


plt.figure(figsize=[20,10])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy','Validation Accuracy'],fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy curves using Dropout Regularization',fontsize=16)
plt.grid()


# In[23]:


model.predict(test_data[[8],:])

