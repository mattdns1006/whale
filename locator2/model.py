import tensorflow as tf
import numpy as np
from operator import mul
import sys
sys.path.append("/home/msmith/misc/tfFunctions")
from layers import *
import tensorflow.contrib.layers as layers
from batchNorm2 import bn

fmp = tf.nn.fractional_max_pool
mp = tf.nn.max_pool
conv = tf.nn.conv2d
dilconv = tf.nn.atrous_conv2d
convUp = tf.nn.conv2d_transpose
af = tf.nn.relu
W = weightVar
B = biasVar

def convolution2d(inTensor,inFeats,outFeats,filterSize,stride=1):
    with tf.name_scope("conv2d"):
        with tf.name_scope("weights"):
            weight = W([filterSize,filterSize,inFeats,outFeats])
        with tf.name_scope("biases"):
            bias = B([outFeats])
        with tf.name_scope("conv"):
            out = conv(inTensor,weight,strides=[1,stride,stride,1],padding='SAME') + bias
    return out

def dilated_convolution2d(inTensor,inFeats,outFeats,filterSize,dilation):
    with tf.name_scope("weights"):
        weight = W([filterSize,filterSize,inFeats,outFeats])
    with tf.name_scope("biases"):
        bias = B([outFeats])
    with tf.name_scope("conv"):
        out = tf.nn.atrous_conv2d(value=inTensor,filters=weight,rate=dilation,padding='SAME') + bias
    return out 

def dense(X,nIn,nOut):
    wLin1 = weightVar([nIn,nOut])
    bLin1 = biasVar([nOut])
    dense = tf.matmul(X,wLin1) + bLin1
    return dense

def model0(x,is_training,nLayers=4,initFeats=16,featsInc=0,nDown=8,filterSize=3):
    print(x.get_shape().as_list())

    shortcut = convolution2d(x,3,initFeats,5,stride=8)
    shortcut = bn(shortcut,is_training,name="bnshort")
    shortcut = af(shortcut)

    X = convolution2d(x,3,initFeats,3,stride=1)
    X = bn(X,is_training,name="bnIn0")
    X = af(X)
    X = convolution2d(X,initFeats,initFeats,3,stride=2)
    X = bn(X,is_training,name="bnIn1")
    X = af(X)
    X = convolution2d(X,initFeats,initFeats,3,stride=1)
    X = bn(X,is_training,name="bnIn2")
    X = af(X)
    X = convolution2d(X,initFeats,initFeats,3,stride=2)
    X = bn(X,is_training,name="bnIn3")
    X = af(X)
    X = convolution2d(X,initFeats,initFeats,3,stride=2)
    X = bn(X,is_training,name="bnIn4")
    X = shortcut + X
    X = af(X)
    dilation = 2
    for layerNo in range(nDown):
        if layerNo == 0:
            inFeats = initFeats 
            outFeats = initFeats + featsInc
        else:
            inFeats = outFeats 
            outFeats = outFeats + featsInc 
        with tf.variable_scope("shortcut_{0}".format(layerNo)):
            shortcut = convolution2d(X,inFeats,outFeats,1)
        with tf.variable_scope("conv_{0}".format(layerNo)):
            X = convolution2d(X,inFeats,outFeats,filterSize)
            X = bn(X,is_training,name="bnconv_{0}".format(layerNo))
            shortcut2 = X
        with tf.variable_scope("dilconv_{0}".format(layerNo)):
            X = dilated_convolution2d(X,outFeats,outFeats,filterSize=2,dilation=dilation)
            X = af(bn(X+shortcut+shortcut2,is_training,name="bndconv_{0}".format(layerNo)))
            print(X.get_shape().as_list(),dilation)

    X1 = convolution2d(X,outFeats,3,filterSize)
    out = X1 # output 0,1
    print(out.get_shape().as_list(),dilation)
    return out

if __name__ == "__main__":
    import loadData, pdb
    sf = 256
    inSize = [sf,sf]
    bs = 20
    X, Y, path = loadData.read(csvPath="csvs/test.csv",batchSize=bs,inSize=inSize,outSize=inSize,shuffle=True)
    is_training = tf.placeholder(tf.bool,name="is_training")
    yPred = model0(X,is_training=is_training)
    merged = tf.summary.merge_all()
    count = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        while True:
            summary,y_ = sess.run([merged,yPred],feed_dict={is_training.name:True})
            count += y_.shape[0]
            print(count,y_.shape)


