import tensorflow as tf
import numpy as np
import pandas as pd

import cv2
import os, pdb, sys, glob
from tqdm import tqdm

sys.path.insert(0,"/home/msmith/misc/tfFunctions/")
sys.path.insert(0,"/home/msmith/misc/py/")
from layers import *
from hStackBatch import hStackBatch

from loadData import dataGenerator

bS  = 10 # Batchsize

## HyperParamter defaults
h, w, c = 210, 350, 3
egPath = glob.glob("../imgs/*/head_*")[0]

nClasses = pd.read_csv("../trainCV.csv").label.max()

# Variables 
x = tf.placeholder(tf.float32, shape=[None,h,w,c])
y = tf.placeholder(tf.float32, shape=[None,nClasses])
keepProb = tf.placeholder(tf.float32)


weights = {}
biases = {}
feats = 16 
featsInc = 16
kS = 3


for i in range(6):
    if i == 0:
        inputFeats = c
        outputFeats = feats
    else: 
        inputFeats = outputFeats
        outputFeats = inputFeats + featsInc

    weights[i] = weightVar([kS, kS, inputFeats, outputFeats])
    biases[i] = biasVar([outputFeats])

hconv1 = tf.nn.relu(bn(feats, conv2d(x, weights[0]) + biases[0]))
hconv1 = mp(hconv1,kS=2,stride=2)

feats += featsInc

hconv2 = tf.nn.relu(bn(feats,conv2d(hconv1, weights[1]) + biases[1]))
hconv2 = mp(hconv2,kS=2,stride=2)

feats += featsInc

hconv3 = tf.nn.relu(bn(feats,conv2d(hconv2, weights[2]) + biases[2]))
hconv3 = mp(hconv3,kS=2,stride=2)

feats += featsInc

hconv4 = tf.nn.relu(bn(feats, conv2d(hconv3, weights[3]) + biases[3]))
hconv4 = mp(hconv4,kS=2,stride=2)

feats += featsInc

hconv5 = tf.nn.relu(bn(feats,conv2d(hconv4, weights[4]) + biases[4]))
hconv5 = mp(hconv5,kS=2,stride=2)

feats += featsInc

hconv6 = tf.nn.relu(bn(feats,conv2d(hconv5, weights[5]) + biases[5]))
hconv6 = mp(hconv6,kS=2,stride=2)

sizeBeforeReshape = hconv6.get_shape().as_list()

nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*outputFeats
flatten = tf.reshape(hconv6, [-1, nFeats])

nLin = 512
wLin1 = weightVar([nFeats,nLin])
bLin1 = biasVar([nLin])
fc1 = tf.nn.relu(tf.matmul(flatten,wLin1) + bLin1)
fc1Drop = tf.nn.dropout(fc1,keepProb)

wLin2 = weightVar([nLin,nClasses])
bLin2 = biasVar([nClasses])

yPred = tf.nn.softmax(tf.matmul(fc1Drop,wLin2) + bLin2)

for l in [hconv1,hconv2,hconv3,hconv4,hconv5,flatten,fc1,yPred]:
    print(l.get_shape())

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yPred), reduction_indices=[1]))
trainStep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(yPred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

load = 0
display = 1
if display == 1:
    import matplotlib.pyplot as plt

    def displayBatch(X,Y,names):
        bs = X.shape[0]


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(init)

    if load == 1:
        saver.restore(sess,"model.chk")

    train = dataGenerator(trainOrTest="train",bS=bS,inputSize=(w,h,c)).generator()
    test = dataGenerator(trainOrTest="test",bS=bS,inputSize=(w,h,c)).generator()

    while True:
        maLossesTrain = []
        maLossesTest = []

        for i in tqdm(range(200),"training"):
            X,Y = next(train)
            pdb.set_trace()
            _, loss = sess.run([trainStep,cross_entropy], feed_dict={x: X, y: Y, keepProb: 0.5})
            maLossesTrain.append(loss)

        meanLoss = np.array(maLossesTrain).mean()
        print("Mean average loss at {0} = {1}".format(i,meanLoss))

        for i in tqdm(range(50),"testing"):
            X,Y = next(test)
            loss = sess.run([cross_entropy], feed_dict={x: X, y: Y, keepProb: 1.0})
            maLossesTest.append(loss)

        meanLossTest = np.array(maLossesTest).mean()
        print("Mean average test loss at {0} = {1}".format(i,meanLossTest))

    saver.save(sess,"model.chk")


        


