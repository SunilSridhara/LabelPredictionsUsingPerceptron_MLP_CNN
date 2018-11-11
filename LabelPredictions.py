
# coding: utf-8

# ##  Adding necessary libraries

# In[4]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import tensorflow as tf
from pylab import rcParams
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras import optimizers
from keras.models import Model


# In[ ]:




# In[ ]:




# ## Reading the data files

# In[5]:

data = pd.read_csv('data.csv',encoding='latin-1')
data.converse = data.converse.astype(str)


# In[ ]:




# In[6]:

data.info()


# In[7]:

data.describe()


# In[8]:

data.head()


# ## Checking labels and their frequencies

# In[9]:

classes = np.unique(data['categories'],return_counts=True)


# In[10]:

print(classes)


# In[11]:

LABELS = ["PRES", "APP","MISC", "ASK", "LAB","JUNK"]
count_classes = pd.value_counts(data['categories'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction category distribution")
plt.xticks(range(6), LABELS)
plt.xlabel("Categories")
plt.ylabel("Frequency")


# In[12]:

count_classes


# ### Removing the "field" column

# In[13]:

dt = data.drop('field',1)
dt = dt[pd.notnull(dt['categories'])]  #### remove all null values


# In[14]:

dt.shape


# In[ ]:




# In[15]:

data.shape


# In[ ]:




# ### Converting unstructured test to structured form

# In[16]:

from sklearn.model_selection import train_test_split
RANDOM_SEED = 123

X_train, X_test = train_test_split(dt, test_size=0.3, random_state=RANDOM_SEED)
print(X_train.head(5))
print('\n')
print(X_test.head(5))


# In[ ]:




# In[17]:

count = pd.value_counts(X_train['categories'], sort = True)
count


# In[ ]:




# In[18]:

count = pd.value_counts(X_test['categories'], sort = True)
count


# In[19]:

unique_labels = list(dt.categories.unique())
y_test_act = np.array([unique_labels.index(i) for i in X_test.categories])
y_test_act[:50]


# In[ ]:




# In[20]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000) #Tokenizer is used to tokenize text
tokenizer.fit_on_texts(dt.converse) #Fit this to our corpus


# In[ ]:




# In[21]:

x_train = tokenizer.texts_to_sequences(X_train.converse) #'text to sequences converts the text to a list of indices
x_train = pad_sequences(x_train, maxlen=50) #pad_sequences makes every sequence a fixed size list by padding with 0s 
x_test = tokenizer.texts_to_sequences(X_test.converse) 
x_test = pad_sequences(x_test, maxlen=50)

x_train.shape, x_test.shape # Check the dimensions of x_train and x_test 


# In[22]:

vocab_size = len(tokenizer.word_index)+1


# In[ ]:




# ### Preparing target vectors for the network

# In[23]:

from keras.utils.np_utils import to_categorical # This convers the 'categories' to one-hot vectors(Dummies)

unique_labels = list(dt.categories.unique()) 
y_train = np.array([unique_labels.index(i) for i in X_train.categories]) # Convert the word labels to indeces
y_train = to_categorical(y_train) # Dummify the labels
y_test = np.array([unique_labels.index(i) for i in X_test.categories])
y_test = to_categorical(y_test)
print(y_test[:5,])


# In[ ]:




# In[24]:

import keras.backend as K # This 'K' can be used to create user defined functions in keras

# Define a custom function in keras to compute recall.
# Arguments:
# y_true - Actual labels
# y_pred - Predicted labels
def recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    PP = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (PP + K.epsilon())
    return recall


# In[ ]:




# # Building Models

# ## Perceptron Model

# In[25]:

# Building a perceptron model
Perceptron_m = Sequential() #defining containers
Perceptron_m.add(Dense(6, input_shape=(50,), activation='sigmoid')) #adding Dense layer, Dense=1, 1 layer and 1 neuron at a time, shape=2  means two attributes red and blue, 
Perceptron_m.compile(optimizer='adam',  #adam is a variant of gradient descent others sgd
      loss='binary_crossentropy', # this is loss function
      metrics=['accuracy'])    # metrics used is accuracy


# In[ ]:




# In[26]:

# Training a perceptron model on the 2D data
Perceptron_m.fit(x_train, y_train, epochs=20)


# In[27]:

# Predicting the class category for test data
preds = Perceptron_m.predict_classes(x_test).reshape(-1,).astype(np.int8)


# In[28]:

print(preds)


# In[29]:

from sklearn.metrics import classification_report

target_names = ['PRES', 'APP', 'MISC','ASK', 'LAB','JUNK']
print(classification_report(y_test_act, preds, target_names=target_names))


# ### MLP Model

# In[30]:

MLP_m = Sequential()
MLP_m.add(Dense(6, input_shape=(50,), activation='sigmoid'))
MLP_m.add(Dense(6, activation='sigmoid'))
SGD = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
MLP_m.compile(optimizer=SGD,
      loss='binary_crossentropy',
      metrics=['accuracy'])


# In[31]:

MLP_m.summary()


# In[ ]:




# In[32]:

MLP_m.fit(x_train, y_train, epochs=20, verbose=2)


# In[ ]:




# In[33]:

preds = MLP_m.predict_classes(x_test).reshape(-1,).astype(np.int8)


# In[ ]:




# In[34]:

print(classification_report(y_test_act, preds, target_names=target_names))


# ### CNN model

# In[35]:

from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Activation


# In[ ]:




# In[36]:

#setting parameters
maxlen = 50
batch_size = 32
embedding_dims = 50
filters = 128
kernel_size = 5
hidden_dims = 10
epochs = 2


# In[ ]:




# In[37]:

CNN_m = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
CNN_m.add(Embedding(vocab_size,
                    embedding_dims,
                    input_length=maxlen))
#CNN_m.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
CNN_m.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
# we use max pooling:
CNN_m.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
CNN_m.add(Dense(hidden_dims,activation='relu'))
#CNN_m.add(Dropout(0.2))
#CNN_m.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
CNN_m.add(Dense(6))
CNN_m.add(Activation('sigmoid'))


# In[ ]:




# In[38]:

CNN_m.summary()


# In[39]:

CNN_m.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:




# In[40]:

CNN_m.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
validation_data=(x_test, y_test))


# In[41]:

preds = CNN_m.predict_classes(x_test).reshape(-1,).astype(np.int8)


# In[ ]:




# In[42]:

print(classification_report(y_test_act, preds, target_names=target_names))


# In[ ]:



