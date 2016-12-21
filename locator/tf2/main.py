import tensorflow as tf
import pandas as pd
from numpy import random as rng
import os, glob
import matplotlib.pyplot as plt

from loadData import loadData
from model import model1

def show(X,Y,yPred, name= "",figsize=(10,10)):
    bs, h, w, c = X.shape 
    X = X.reshape(bs*h,w,c)
    bs, h, w, c = Y.shape 
    Y = Y.reshape(bs*h,w,c)
    yPred = yPred.reshape(bs*h,w,c)
    plt.figure(figsize=figsize) 
    plt.subplot(131)
    plt.imshow(X)
    plt.subplot(132)
    plt.imshow(Y)
    plt.subplot(133)
    plt.imshow(yPred)
    plt.savefig("models/images/{0}.jpg".format(name))
    plt.close()

def mse(y,yPred):
    return tf.reduce_mean(tf.square(tf.sub(y,yPred)))

def trainer(loss,learningRate,momentum=0.95):
        return tf.train.AdamOptimizer(learningRate).minimize(loss)

if __name__ == "__main__":
	import ipdb
	inShape = [800,1100,3]
	outShape = [50,69,3]
        batchSize = 4
        nFeatsInc = 0 
        nFeats = 32
        nEpochs = 10
        load = 0
        modelName = "models/model1.tf"

        def nodes(trOrTe):
            if trOrTe == "train":
                path = "train.csv"
            else:
                path = "test.csv"
            x, y, paths = loadData(path,inShape,outShape,batchSize)
            yPred = model1(x,nFeats,nFeatsInc)
            loss = mse(y,yPred)
            if trOrTe == "train":
                learningRate = tf.placeholder(tf.float32)
                train = trainer(loss,learningRate,momentum=0.95)
                return x, y, yPred, paths, loss, learningRate, train
            else:
                return x, y, yPred, paths,loss


        lr = 0.03
        
        for epoch in range(nEpochs):
            if epoch > 1 or load == 1:
                tf.reset_default_graph()


            x, y, yPred, paths, loss, learningRate, train = nodes("train")
            saver = tf.train.Saver()

            with tf.Session() as sess:
                    if load == 1:
                        saver.restore(sess,modelName)
                    else:
                        init = tf.initialize_all_variables()
                        sess.run(init)

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                    count = 0
                    for i in xrange(10000):
                        _, loss_, x_, y_, yPred_, paths_ = sess.run([train,loss,x,y,yPred,paths],feed_dict={learningRate:lr})
                        if count > 10000:
                            lr /= 2.0
                        if count % 100 == 0 :
                            show(x_,y_,yPred_,count)
                        if count % 2000 == 0 and count > 0:
                            saver.save(sess,modelName)
                        count += batchSize 
                        print(count,loss_)
                    saver.save(sess,modelName)
                    sess.close()



