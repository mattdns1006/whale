import tensorflow as tf
import numpy as np
from operator import mul
import sys
sys.path.append("/home/msmith/misc/tfFunctions")
from layers import *
import tensorflow.contrib.layers as layers
from batchNorm2 import bn

#bn = layers.batch_norm
fmp = tf.nn.fractional_max_pool
mp = tf.nn.max_pool
conv = tf.nn.conv2d
dilconv = tf.nn.atrous_conv2d
convUp = tf.nn.conv2d_transpose
af = tf.nn.relu
W = weightVar
B = biasVar

def convolution2d(inTensor,size,inFeats,outFeats):
    with tf.name_scope("conv2d"):
        with tf.name_scope("weights"):
            weight = W([size,size,inFeats,outFeats])
        with tf.name_scope("biases"):
            bias = B([outFeats])
        with tf.name_scope("conv"):
            filtered = conv(inTensor,weight,strides=[1,1,1,1],padding='SAME') + bias
    return filtered

def dilated_convolution2d(inTensor,fS,inFeats,outFeats,dilation):
    with tf.name_scope("weights"):
        weight = W([fS,fS,inFeats,outFeats])
    with tf.name_scope("biases"):
        bias = B([outFeats])
    with tf.name_scope("conv"):
        filtered = tf.nn.atrous_conv2d(inTensor,weight,dilation,padding='SAME') + bias
    return filtered

def denseBlock(x,inFeats,outFeats,layerNo,is_training):
    def growth(inTensor,toAdd,growthNumber):
        with tf.name_scope("growth_{0}".format(growthNumber)):
            filtered = convolution2d(inTensor,inFeats,outFeats)
            with tf.name_scope("add"):
                out = af(toAdd + filtered)
        return out
    x1 = growth(inTensor=x,toAdd=x,growthNumber=1)
    x10 = tf.add(x1,x)
    x2 = growth(inTensor=x1,toAdd=x10,growthNumber=2)
    x210 = tf.add(x2,x10)
    x3 = growth(inTensor=x2,toAdd=x210,growthNumber=3)
    x3210 = tf.add(x3,x210)
    x4 = growth(inTensor=x3,toAdd=x3210,growthNumber=4)
    return x4

def dense(X,nIn,nOut):
    wLin1 = weightVar([nIn,nOut])
    bLin1 = biasVar([nOut])
    dense = tf.matmul(X,wLin1) + bLin1
    return dense

def model0(x,is_training,nLayers=4,initFeats=16,featsInc=16,nDown=8,filterSize=3):
    X = convolution2d(x,filterSize,3,initFeats)
    X = bn(X,is_training,name="bn")
    X = af(X)
    for layerNo in range(nDown):
        if layerNo == 0:
            inFeats = initFeats 
            outFeats = initFeats + featsInc
        else:
            inFeats = outFeats 
            outFeats = initFeats
        with tf.variable_scope("layerNo_{0}".format(layerNo)):
            X = convolution2d(X,filterSize,inFeats,outFeats)
            X = bn(X,is_training,name="bn_{0}".format(layerNo))
            X = af(X)

            if layerNo == nDown - 1:
                break
	    X = tf.nn.max_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],padding = "SAME",name="mp_".format(layerNo))
	
    nFeats = reduce(mul,X.get_shape().as_list()[1:])
    X = tf.reshape(X,[-1,nFeats])

    nDense = 2
    for layerNo in xrange(nDense):
        if layerNo == 0:
            nIn = nFeats
        else:
            nIn = 128
        X = af(bn(dense(X,nIn,128), name = "bnLin_{0}".format(layerNo),is_training=is_training))
        
    out = tf.nn.sigmoid(dense(X,128,4)) # output 0,1
    return out

if __name__ == "__main__":
    import loadData, pdb
    sf = 256
    inSize = [sf,sf]
    img, x1,y1,x2,y2,w,h = loadData.read(csvPath="train.csv",batchSize=4,inSize=inSize,shuffle=True)
    X = tf.placeholder(tf.float32,shape=[None,sf,sf,1])
    is_training = tf.placeholder(tf.bool,name="is_training")
    Y = model0(X,is_training=is_training,nDown=8,initFeats=16)
    pdb.set_trace()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in xrange(10):
            x = np.random.rand(1,256,256,1)
            summary,y_ = sess.run([merged,Y],feed_dict={X:x,is_training.name:True})
            print(y_.shape)
            if i == 9:
                pdb.set_trace()


