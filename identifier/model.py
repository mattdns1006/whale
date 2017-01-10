import tensorflow as tf
import sys
sys.path.insert(0,"/home/msmith/misc/tfFunctions/")

from layers import *
#import tensorflow.contrib.layers as layers
#bn = layers.batch_norm

from batchNorm import batch_norm as bn
af = tf.nn.relu


def model1(x,inDims,nClasses,nFeatsInit,nFeatsInc,keepProb,training):
	bs, h, w, c = inDims
	weights = {}
	biases = {}
	feats = nFeatsInit
	featsInc = nFeatsInc 
	kS = 3
        mpkS = 3

	for i in range(9):
	    if i == 0:
		inputFeats = c
		outputFeats = feats
	    else: 
		inputFeats = outputFeats
		outputFeats = inputFeats + featsInc

	    weights[i] = weightVar([kS, kS, inputFeats, outputFeats])
	    biases[i] = biasVar([outputFeats])

	    print(x.get_shape())
            x = af(bn((conv2d(x, weights[i]) + biases[i]),training=training,name="bn_{0}".format(i)))
            x = mp(x,kS=mpkS,stride=2)

	    feats += featsInc

	sizeBeforeReshape = x.get_shape().as_list()

	nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*sizeBeforeReshape[3]

	flatten = tf.reshape(x, [-1, nFeats])
        flatten = tf.nn.dropout(flatten,keepProb)

	nLin1 = 128 
	wLin1 = weightVar([nFeats,nLin1])
	bLin1 = biasVar([nLin1])
	linear = af(tf.matmul(flatten,wLin1) + bLin1)

	nLin2 = nClasses
	wLin2 = weightVar([nLin1,nLin2])
	bLin2 = biasVar([nLin2])
	yPred = tf.matmul(linear,wLin2) + bLin2

	return yPred

def model2(x,inDims,nClasses,nFeatsInit,nFeatsInc,training):
	bs, h, w, c = inDims
	weights = {}
	biases = {}
	feats = nFeatsInit
	featsInc = nFeatsInc 
	kS = 3

        def block(inTensor,inputFeats,outputFeats,kS):
            w1 = weightVar([kS, kS, inputFeats, outputFeats])
            b1 = biasVar([outputFeats])
            print(w1.get_shape())
            print(b1.get_shape())
            w2 = weightVar([kS, kS, outputFeats, outputFeats])
            b2 = biasVar([outputFeats])
            print(w2.get_shape())
            print(b2.get_shape())
            hconv = af(bn((conv2d(inTensor, w1) + b1),training=training))
            hconv = af(bn((conv2d(hconv, w2) + b2),training=training))
            block = mp(hconv,kS=2,stride=2)
            return block
        
        nOut = nFeatsInit
        block1 = block(x,3,nOut,kS)
        nIn = nOut
        nOut = nFeatsInit + nFeatsInc

        block2 = block(block1,nIn,nOut,kS)
        nIn = nOut
        nOut += nFeatsInc

        block3 = block(block2,nIn,nOut,kS)
        nIn = nOut
        nOut += nFeatsInc

        block4 = block(block3,nIn,nOut,kS)
        nIn = nOut
        nOut += nFeatsInc

        block5 = block(block4,nIn,nOut,kS)
        nIn = nOut
        nOut += nFeatsInc

        block6 = block(block5,nIn,nOut,kS)
        nIn = nOut
        nOut += nFeatsInc

        block7 = block(block6,nIn,nOut,kS)

	sizeBeforeReshape = block7.get_shape().as_list()

	nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*sizeBeforeReshape[3]
	flatten = tf.reshape(block7, [-1, nFeats])

	nLin = nClasses
	wLin1 = weightVar([nFeats,nLin])
	bLin1 = biasVar([nLin])
	yPred = tf.matmul(flatten,wLin1) + bLin1
        yPred = bn(yPred)
        return yPred


