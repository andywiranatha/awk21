#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset = pd.read_csv("E202-COMP7117-TD01-00 - classification.csv", delimiter = ",")
print(dataset.head())
from sklearn import preprocessing


# In[2]:


pickedDataSet = dataset[["volatile acidity", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
print(pickedDataSet.head())
output = dataset[["quality"]]
print(output.head())


# In[3]:


i=0
for i in range (len(pickedDataSet[["free sulfur dioxide"]])):
    if pickedDataSet[["free sulfur dioxide"]].values[i] == 'Low':
        pickedDataSet.at[i, 'free sulfur dioxide'] = 1 
    elif pickedDataSet[["free sulfur dioxide"]].values[i] == 'Medium':
        pickedDataSet.at[i, 'free sulfur dioxide'] = 2
    elif pickedDataSet[["free sulfur dioxide"]].values[i] == 'High':
        pickedDataSet.at[i, 'free sulfur dioxide'] = 3
    else :
        pickedDataSet.at[i, 'free sulfur dioxide'] = 0
#     print(pickedDataSet[["SpecialDay"]].values[i])
        
print(pickedDataSet[["free sulfur dioxide"]])


# In[4]:


i=0
for i in range (len(pickedDataSet[["density"]])):
    if pickedDataSet[["density"]].values[i] == 'Low':
        pickedDataSet.at[i, 'density'] = 1 
    elif pickedDataSet[["density"]].values[i] == 'Medium':
        pickedDataSet.at[i, 'density'] = 2
    elif pickedDataSet[["density"]].values[i] == 'High':
        pickedDataSet.at[i, 'density'] = 3
    elif pickedDataSet[["density"]].values[i] == 'Very High':
        pickedDataSet.at[i, 'density'] = 0
#     print(pickedDataSet[["SpecialDay"]].values[i])
        
print(pickedDataSet[["density"]])


# In[6]:


i=0
for i in range (len(pickedDataSet[["pH"]])):
    if pickedDataSet[["pH"]].values[i] == 'Very Basic':
        pickedDataSet.at[i, 'pH'] = 1 
    elif pickedDataSet[["pH"]].values[i] == 'Normal':
        pickedDataSet.at[i, 'pH'] = 2
    elif pickedDataSet[["pH"]].values[i] == 'Very Acidic':
        pickedDataSet.at[i, 'pH'] = 3
    else :
        pickedDataSet.at[i, 'pH'] = 0
#     print(pickedDataSet[["SpecialDay"]].values[i])
        
print(pickedDataSet[["pH"]])


# In[7]:


normalized_data = preprocessing.normalize(pickedDataSet)
print(normalized_data)


# In[8]:


from sklearn.decomposition import PCA


# In[10]:


pca = PCA(n_components=4)
principalComponents = pca.fit_transform(normalized_data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])


# In[11]:


print(principalDf)


# In[33]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# In[34]:


def load():

    feat = principalDf

    target = output

    scaler = MinMaxScaler()
    feat = scaler.fit_transform(feat)

    encoder = OneHotEncoder(sparse=False)
    target = encoder.fit_transform(target)

    return feat, target


# In[45]:


feat, target = load()

layer = {
    "input" : 4,
    "hidden" : 6,
    "output" : 5
}

weight = {
    "input_hidden" : tf.Variable(tf.random_normal([layer['input'],layer['hidden']])),
    "hidden_output" : tf.Variable(tf.random_normal([layer['hidden'],layer['output']]))
}

bias = {
    "input_hidden" : tf.Variable(tf.random_normal([layer['hidden']])),
    "hidden_output" : tf.Variable(tf.random_normal([layer['output']]))
}

feat_input = tf.placeholder(tf.float32, [None, layer['input']])
target_input = tf.placeholder(tf.float32, [None, layer['output']])

def feed_forward(datas):
    w1 = tf.matmul(datas, weight['input_hidden']) + bias['input_hidden']
    w1 = tf.nn.sigmoid(w1)

    w2 = tf.matmul(w1,weight['hidden_output']) + bias['hidden_output']
    return tf.nn.sigmoid(w2)

epoch = 5000

prediction = feed_forward(feat_input)

loss = tf.reduce_mean(0.5*( target_input - prediction )**2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
train = optimizer.minimize(loss)

train_data, test_data, train_target, test_target = train_test_split(feat, target, test_size = 0.1)
train_data, validation_data, train_target, validation_target = train_test_split(feat,target,test_size = 0.2)

#train_data nya jadi nya 0.72 => 0.7
#validatio_data nya jadinya 0.18 => 0.2
#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, epoch+1):
        train_dict = {
            feat_input : train_data,
            target_input : train_target
        }

        sess.run(train, feed_dict = train_dict)

        error = sess.run(loss,feed_dict = train_dict)

        if i % 100 == 0:
            print("Iteration {} error : {}".format(i, error))
            
        if i % 500 == 0:
            validation_dict = {
                feat_input : validation_data,
                target_input : validation_target
            }

            sess.run(train, feed_dict = validation_dict)
            validationError = sess.run(loss,feed_dict = validation_dict)
#             print(validationError)
            
            if(i==500):
                lowValidationError = validationError
                w=open("lowValidationError.txt", "w")
                w.write(str(lowValidationError))
                w.close()
            if validationError < lowValidationError:
                lowValidationError = validationError
                w=open("lowValidationErrorr.txt", "w")
                w.write(str(lowValidationError))
                w.close()
            print("Iteration {} lowValidationError : {}".format(i, lowValidationError))
            
            
            
    matches = tf.equal(tf.argmax(target_input,axis = 1), tf.argmax(prediction, axis = 1))

    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    test_dict = {
        feat_input : test_data,
        target_input : test_target
    }

    print("Accuracy : {}%".format(sess.run(accuracy, feed_dict=test_dict) * 100))
    r=open("lowValidationError.txt","r")
    if r.mode =='r':
        contents = r.read()
    print(contents)
    r.close()


# In[ ]:




