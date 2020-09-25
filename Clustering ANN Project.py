#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset = pd.read_csv("E202-COMP7117-TD01-00 - clustering.csv", delimiter = ",")
print(dataset.head())
from sklearn import preprocessing


# In[2]:


pickedDataSet = dataset[["SpecialDay", "VisitorType", "Weekend", "ProductRelated_Duration", "ExitRates"]]
print(pickedDataSet.head())
print(pickedDataSet.SpecialDay)
print(len(pickedDataSet.SpecialDay))


# In[3]:


i=0
for i in range (len(pickedDataSet[["SpecialDay"]])):
    if pickedDataSet[["SpecialDay"]].values[i] == 'LOW':
        pickedDataSet.at[i, 'SpecialDay'] = 0
        
    elif pickedDataSet[["SpecialDay"]].values[i] == 'NORMAL':
        pickedDataSet.at[i, 'SpecialDay'] = 1
    elif pickedDataSet[["SpecialDay"]].values[i] == 'HIGH':
        pickedDataSet.at[i, 'SpecialDay'] = 2
#     print(pickedDataSet[["SpecialDay"]].values[i])
        
print(pickedDataSet.at[6, 'SpecialDay'])

pickedDataSet.SpecialDay.dtypes


# In[4]:


print(pickedDataSet.VisitorType)


# In[5]:


i=0
for i in range (len(pickedDataSet[["VisitorType"]])):
    if pickedDataSet[["VisitorType"]].values[i] == 'Other':
        pickedDataSet.at[i, 'VisitorType'] = 0
    elif pickedDataSet[["VisitorType"]].values[i] == 'New_Visitor':
        pickedDataSet.at[i, 'VisitorType'] = 1
    elif pickedDataSet[["VisitorType"]].values[i] == 'Returning_Visitor':
        pickedDataSet.at[i, 'VisitorType'] = 2
    print(pickedDataSet[["VisitorType"]].values[i])
    


# In[6]:



pickedDataSet[["Weekend"]] = pickedDataSet[["Weekend"]].astype(int)
print(pickedDataSet[["Weekend"]])


# In[7]:


print(pickedDataSet[["Weekend"]])


# In[8]:


print(pickedDataSet)


# In[9]:


normalized_data = preprocessing.normalize(pickedDataSet)
print(normalized_data)


# In[10]:


from sklearn.decomposition import PCA


# In[11]:


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(normalized_data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])


# In[12]:


print(principalDf)
print(len(principalDf))


# In[13]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt


# In[14]:


class SOM:
    def __init__(self, width, height, input_dimension):
        self.width = width
        self.height = height
        self.input_dimension = input_dimension

        self.weight = tf.Variable(tf.random_normal([width * height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]

        self.bmu = self.getBMU()

        self.update_weight = self.update_neigbours()

    def getBMU(self):
        #Best Matching Unit

        #Eucledian distance
        square_distance = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_sum(square_distance, axis=1))

        #Get BMU index
        bmu_index = tf.argmin(distance)
        #Get the position
        bmu_position = tf.to_float([tf.div(bmu_index,self.width), tf.mod(bmu_index, self.width)])
        return bmu_position

    def update_neigbours(self):

        learning_rate = 0.1

        #Formula calculate sigma / radius
        sigma = tf.to_float(tf.maximum(self.width, self.height) / 2)

        #Eucledian Distance between BMU and location
        square_difference = tf.square(self.bmu - self.location)
        distance = tf.sqrt(tf.reduce_sum(square_difference,axis=1))

        #Calculate Neighbour Strength based on formula
        # NS = tf.exp((- distance ** 2) /  (2 * sigma ** 2))
        NS = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))

        #Calculate rate before reshape
        rate = NS * learning_rate

        #Reshape to [width * height, input_dimension]
        rate_stacked = tf.stack([tf.tile(tf.slice(rate,[i],[1]), [self.input_dimension]) 
            for i in range(self.width * self.height)])

        #Calculate New Weight
        new_weight = self.weight + rate_stacked * (self.input - self.weight)

        return tf.assign(self.weight, new_weight)

    def train(self, dataset, epoch):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #training
            for i in range(epoch+1):
                for data in dataset:
                    dictionary = {
                        self.input : data
                    }

                    sess.run(self.update_weight,feed_dict=dictionary)

            #assign clusters
            location = sess.run(self.location)
            weight = sess.run(self.weight)

            clusters = [[] for i in range(self.height)]

            for i, loc in enumerate(location):
                clusters[int(loc[0])].append(weight[i])

            self.clusters = clusters


# In[15]:


input_dimension = len(principalComponents[0])
print(input_dimension)
print(principalComponents.shape)


# In[16]:


epoch = 5000
som = SOM(15,15,input_dimension)
som.train(principalComponents,epoch)
plt.imshow(som.clusters)
plt.show()


# In[ ]:




