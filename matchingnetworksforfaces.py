from numpy import load
from keras.models import Model
from keras.layers import Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras
from matchnn import MatchCosine
import numpy as np


def packslice( data_pack, numsamples):
    
    classes_per_set = 5
    samples_per_class = 1

    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []
    
    for i in range(numsamples):
        slice_x = np.zeros((n_samples+1,16,8,1))
        slice_y = np.zeros((n_samples,))
        
        ind = 0
        pinds = np.random.permutation(n_samples)
        classes = np.random.choice(data_pack.shape[0],classes_per_set,False) 
        
        x_hat_class = np.random.randint(classes_per_set) 
        
        for j, cur_class in enumerate(classes): 
            example_inds = np.random.choice(data_pack.shape[1],samples_per_class,False)
            
            for eind in example_inds:
                slice_x[pinds[ind],:,:,:] = data_pack[cur_class][eind]
                slice_y[pinds[ind]] = j
                ind += 1
            
            if j == x_hat_class:
                slice_x[n_samples,:,:,:] = data_pack[cur_class][np.random.choice(data_pack.shape[1])]
                target_y = j

        support_cacheX.append(slice_x)
        support_cacheY.append(keras.utils.to_categorical(slice_y,classes_per_set))
        target_cacheY.append(keras.utils.to_categorical(target_y,classes_per_set)[0])
        
    return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)


data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))



#Encoding labels as integers:
for i in range(len(testy)):
  if testy[i]=='ben_afflek':
    testy[i] = 1
  if testy[i]=='elton_john':
    testy[i] = 2
  if testy[i]=='madonna':
    testy[i] = 3
  if testy[i]=='mindy_kaling':
    testy[i] = 4
  if testy[i]=='jerry_seinfeld':
    testy[i] = 0

for i in range(len(trainy)):
  if trainy[i]=='ben_afflek':
    trainy[i] = 1
  if trainy[i]=='elton_john':
    trainy[i] = 2
  if trainy[i]=='madonna':
    trainy[i] = 3
  if trainy[i]=='mindy_kaling':
    trainy[i] = 4
  if trainy[i]=='jerry_seinfeld':
    trainy[i] = 0



shuffle_classes_test=np.arange(testX.shape[0])
np.random.shuffle(shuffle_classes_test)
testX=testX[shuffle_classes_test]


indexes = {"train": 0, "val": 0}
datasets = {"train": trainX, "val": testX}
datasets_cache = {"train": packslice(datasets["train"],300),
                      "val": packslice(datasets["val"],50)}



#one hot encoding the interger labels
trainy = keras.utils.np_utils.to_categorical(trainy, num_classes=5)
testy = keras.utils.np_utils.to_categorical(testy, num_classes=5)


bsize = 32 # batch size
classes_per_set = 5 # classes per set or 5-way
samples_per_class = 1 # samples per class 1-shot

conv1 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm1 = BatchNormalization()
mpool1 = MaxPooling2D((2,2),padding='same')
conv2 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm2 = BatchNormalization()
mpool2 = MaxPooling2D((2,2),padding='same')
conv3 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm3 = BatchNormalization()
mpool3 = MaxPooling2D((2,2),padding='same')
conv4 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm4 = BatchNormalization()
mpool4 = MaxPooling2D((2,2),padding='same')
fltn = Flatten()

def convembedding(x):
    x = conv1(x)
    x = bnorm1(x)
    x = mpool1(x)
    x = conv2(x)
    x = bnorm2(x)
    x = mpool2(x)
    x = conv3(x)
    x = bnorm3(x)
    x = mpool3(x)
    x = conv4(x)
    x = bnorm4(x)
    x = mpool4(x)
    x = fltn(x)
    
    return x

numsupportset = samples_per_class*classes_per_set
input1 = Input((numsupportset+1,16,8,1))

modelinputs = []
for lidx in range(numsupportset):
    modelinputs.append(convembedding(Lambda(lambda x: x[:,lidx,:,:,:])(input1)))
targetembedding = convembedding(Lambda(lambda x: x[:,-1,:,:,:])(input1))
modelinputs.append(targetembedding)
supportlabels = Input((numsupportset,classes_per_set))
modelinputs.append(supportlabels)

knnsimilarity = MatchCosine(nway=classes_per_set)(modelinputs)

model = Model(inputs=[input1,supportlabels],outputs=knnsimilarity)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

model.fit([datasets_cache["train"][0],datasets_cache["train"][1]],datasets_cache["train"][2],
          validation_data=[[datasets_cache["val"][0],datasets_cache["val"][1]],datasets_cache["val"][2]],
          epochs=10,batch_size=32,verbose=1)

