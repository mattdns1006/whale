import tensorflow as tf
import numpy as np
import pandas as pd

import cv2
import os, ipdb, sys, glob
from tqdm import tqdm

sys.path.insert(0,"/home/msmith/misc/tfFunctions/")
sys.path.insert(0,"/home/msmith/misc/py/")
from layers import *
from hStackBatch import hStackBatch

from loadData import dataGenerator

bS  = 2 # Batchsize

## HyperParamter defaults
h, w, c = 420, 700, 3
egPath = glob.glob("../imgs/*/head_*")[0]

nClasses = pd.read_csv("../trainCV.csv").label.max()

# Variables 
x = tf.placeholder(tf.float32, shape=[None,h,w,c])
y = tf.placeholder(tf.float32, shape=[None,nClasses])
keepProb = tf.placeholder(tf.float32)


weights = {}
biases = {}
feats = 32 
featsInc = 16 
kS = 3


for i in range(8):
    if i == 0:
        inputFeats = c
        outputFeats = feats
    else: 
        inputFeats = outputFeats
        outputFeats = inputFeats + featsInc

    weights[i] = weightVar([kS, kS, inputFeats, outputFeats])
    biases[i] = biasVar([outputFeats])

hconv1 = tf.nn.relu(conv2d(x, weights[0]) + biases[0])
hconv1 = mp(hconv1,kS=2,stride=2)

feats += featsInc

hconv2 = tf.nn.relu(conv2d(hconv1, weights[1]) + biases[1])
hconv2 = mp(hconv2,kS=2,stride=2)

feats += featsInc

hconv3 = tf.nn.relu(conv2d(hconv2, weights[2]) + biases[2])
hconv3 = mp(hconv3,kS=2,stride=2)

feats += featsInc

hconv4 = tf.nn.relu(conv2d(hconv3, weights[3]) + biases[3])
hconv4 = mp(hconv4,kS=2,stride=2)

feats += featsInc

hconv5 = tf.nn.relu(conv2d(hconv4, weights[4]) + biases[4])
hconv5 = mp(hconv5,kS=2,stride=2)

feats += featsInc

hconv6 = tf.nn.relu(conv2d(hconv5, weights[5]) + biases[5])
hconv6 = mp(hconv6,kS=2,stride=2)

feats += featsInc

hconv7 = tf.nn.relu(conv2d(hconv6, weights[6]) + biases[6])
hconv7 = mp(hconv7,kS=2,stride=2)

feats += featsInc

hconv8 = tf.nn.relu(conv2d(hconv7, weights[7]) + biases[7])
hconv8 = mp(hconv8,kS=2,stride=2)

sizeBeforeReshape = hconv8.get_shape().as_list()

nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*outputFeats
flatten = tf.reshape(hconv8, [-1, nFeats])

nLin = 512
wLin1 = weightVar([nFeats,nLin])
bLin1 = biasVar([nLin])
fc1 = tf.nn.relu((tf.matmul(flatten,wLin1) + bLin1))

wLin2 = weightVar([nLin,nClasses])
bLin2 = biasVar([nClasses])

yPred = tf.nn.softmax(tf.matmul(fc1,wLin2) + bLin2)

for l in [x, hconv1,hconv2,hconv3,hconv4,hconv5,flatten,fc1,yPred]:
    print(l.get_shape())

classWeights = tf.constant(pd.read_csv("../trWeights.csv").whaleID.values,"float32")
wce = tf.reduce_mean(-tf.reduce_sum(classWeights*y*tf.log(yPred), reduction_indices=[1]))
ce = tf.reduce_mean(-tf.reduce_sum(y*tf.log(yPred), reduction_indices=[1]))
trainStep = tf.train.AdamOptimizer(1e-5).minimize(wce)
correct = tf.equal(tf.argmax(yPred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.initialize_all_variables()

load = 0
display = 1
if display == 1:

    import matplotlib.pyplot as plt

    def displayBatch(XY):
        X,Y = XY
        names = train.decodeToName(Y)
        bs = X.shape[0]
        X = X.astype(np.uint8)
        fig = plt.figure()
        nRow = 2
        x = bs/nRow
        y = bs/x
        idx = 0
        ipdb.set_trace()
        for i in range(1,bs+1):
            ax = fig.add_subplot(bs+1,1,i)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.imshow(X[idx])
            ax.set_title(names[idx])
            idx +=1
        plt.show()
        
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(init)

    if load == 1:
        saver.restore(sess,"model.chk")

    for epoch in range(40):
        train = dataGenerator(trainOrTest="train",bS=bS,inputSize=(w,h,c))
        test = dataGenerator(trainOrTest="test",bS=1,inputSize=(w,h,c))
        trGen = train.generator()
        teGen = test.generator()

        maLossesTrain = []
        maLossesTest = []
        maAccTrain = []
        maAccTest = []

        while train.finishedEpoch == 0:
            XY = X,Y = next(trGen)

            #displayBatch(XY)
            _, loss, _yPred, acc = sess.run([trainStep,ce,yPred,accuracy], feed_dict={x: X, y: Y, keepProb: 1.0})
            maLossesTrain.append(loss)
            maAccTrain.append(acc)

        while test.finishedEpoch == 0:
            XY = X,Y = next(teGen)
            loss, acc  = sess.run([ce,accuracy], feed_dict={x: X, y: Y, keepProb: 1.0})
            maLossesTest.append(loss)
            maAccTest.append(acc)

        perf = [np.array(l).mean() for l in [maLossesTrain,maLossesTest,maAccTrain,maAccTest]]

        print("Mean average losses/accs tr,te after epoch {0} = {1}".format(epoch,perf))

    saver.save(sess,"model.chk")


        


