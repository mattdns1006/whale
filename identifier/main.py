import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os, ipdb, sys, glob
from tqdm import tqdm
from model import model1
sys.path.insert(0,"/home/msmith/misc/py/")
from hStackBatch import hStackBatch
from loadData import dataGenerator 

if __name__ == "__main__":
    bS  = 8 # Batchsize

    ## HyperParameter defaults
    h, w, c = 200, 400, 3
    nClasses = 447
    inDims = [None,h,w,c]

    # Define placeholders and model 
    x = tf.placeholder(tf.float32, shape=inDims)
    y = tf.placeholder(tf.float32, shape=[None,nClasses])
    yPred = model1(x,inDims=inDims,nClasses=nClasses)

    ce = tf.reduce_mean(-tf.reduce_sum(y*tf.log(yPred), reduction_indices=[1]))
    trainStep = tf.train.AdamOptimizer(1e-3).minimize(ce)
    correct = tf.equal(tf.argmax(yPred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.initialize_all_variables()

    load = 0
    display = 1
            
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
                _, loss, _yPred, acc = sess.run([trainStep,ce,yPred,accuracy], feed_dict={x: X, y: Y})
                maLossesTrain.append(loss)
                maAccTrain.append(acc)

            while test.finishedEpoch == 0:
                XY = X,Y = next(teGen)
                loss, acc  = sess.run([ce,accuracy], feed_dict={x: X, y: Y})
                maLossesTest.append(loss)
                maAccTest.append(acc)

            perf = [np.array(l).mean() for l in [maLossesTrain,maLossesTest,maAccTrain,maAccTest]]

            print("Mean average losses/accs tr,te after epoch {0} = {1}".format(epoch,perf))

        saver.save(sess,"model.chk")


        


