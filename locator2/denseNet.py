from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
#import tensorflow.contrib.layers as layers
import sys
sys.path.append("/home/msmith/misc/tfFunctions")
from batchNorm2 import bn
from layers import *

fmp = tf.nn.fractional_max_pool
mp = tf.nn.max_pool
dilconv = tf.nn.atrous_conv2d
convUp = tf.nn.conv2d_transpose
af = tf.nn.relu

def weightVar(shape,stddev=0.35):
    nOut = shape[-1]
    nIn = shape[-2]
    n = (nOut + nIn)/2.0
    limit = math.sqrt(3.0/n)
    initial = tf.random_uniform(shape, -limit, limit)
    return tf.Variable(initial)

def biasVar(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

W = weightVar
B = biasVar

def convolution2d(inTensor,inFeats,outFeats,filterSize,stride=1):
    with tf.name_scope("conv2d"):
        with tf.name_scope("w"):
            weight = W([filterSize,filterSize,inFeats,outFeats])
        with tf.name_scope("b"):
            bias = B([outFeats])
        with tf.name_scope("conv"):
            out = tf.nn.conv2d(inTensor,weight,strides=[1,stride,stride,1],padding='SAME') + bias
    return out

def dilated_convolution2d(inTensor,inFeats,outFeats,filterSize,dilation):
    with tf.name_scope("w"):
        weight = W([filterSize,filterSize,inFeats,outFeats])
    with tf.name_scope("b"):
        bias = B([outFeats])
    with tf.name_scope("conv"):
        out = tf.nn.atrous_conv2d(value=inTensor,filters=weight,rate=dilation,padding='SAME') + bias
    return out 


def model0(x,is_training,initFeats=16,featsInc=0,nDown=8,filterSize=3):
    print(x.get_shape())
    with tf.variable_scope("convIn"):
        x1 = af(bn(convolution2d(x,3,initFeats,3,stride=1),is_training=is_training,name="bn_0"))

    dilation = 2
    for block in range(nDown):
        if block == 0:
            inFeats = initFeats 
            outFeats = initFeats + featsInc
        else:
            inFeats = outFeats 
            outFeats = outFeats + featsInc
        with tf.variable_scope("block_down_{0}".format(block)):
	    x1 = af(bn(convolution2d(x1,inFeats,outFeats,3,stride=1),is_training=is_training,name="bn_{0}_0".format(nDown)))
	    #x1 = af(bn(convolution2d(x1,outFeats,outFeats,3,stride=1),is_training=is_training,name="bn_{0}_1".format(nDown)))
	    #x1 = af(bn(dilated_convolution2d(x1,outFeats,outFeats,3,dilation=dilation),is_training=is_training,name="bn_{0}_2".format(nDown)))
	    x1 = tf.nn.max_pool(x1,[1,3,3,1],[1,2,2,1],"SAME")
            dilation += 4
    	    print(x1.get_shape())

    with tf.variable_scope("convOut"):
        out = tf.nn.sigmoid(bn(convolution2d(x1,outFeats,3,3,stride=1),is_training=is_training,name="bn_out"))
        #out = tf.nn.sigmoid(convolution2d(x1,outFeats,3,3,stride=1))
    print(out.get_shape())
    return out 


def main(argv=None):
    import loadData, pdb
    IMAGE_SIZE = 256
    inSize = [IMAGE_SIZE,IMAGE_SIZE]
    bs = 20
    X, Y, path = loadData.read(csvPath="csvs/test.csv",batchSize=bs,inSize=inSize,outSize=inSize,num_epochs=2,shuffle=True)
    is_training = tf.placeholder(tf.bool,name="is_training")
    yPred = model0(X,is_training=is_training)
    merged = tf.summary.merge_all()
    count = 0
    pdb.set_trace()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        while True:
            summary,y_ = sess.run([merged,yPred],feed_dict={is_training.name:True})
            count += y_.shape[0]
            print(count,y_.shape)


if __name__ == "__main__":
    main()
