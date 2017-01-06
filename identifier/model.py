import tensorflow as tf
import sys
sys.path.insert(0,"/home/msmith/misc/tfFunctions/")

from layers import *
from tensorflow.contrib.layers import layers as tfLayers
bn = tfLayers.batch_norm
af = tf.nn.relu



def model1(x,inDims,nClasses,nFeatsInit,nFeatsInc,keepProb,is_training):
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

	hconv1 = af(bn((conv2d(x, weights[0]) + biases[0]),is_training=is_training))
	hconv1 = mp(hconv1,kS=mpkS,stride=2)

	feats += featsInc

	hconv2 = af(bn((conv2d(hconv1, weights[1]) + biases[1]),is_training=is_training))
	hconv2 = mp(hconv2,kS=mpkS,stride=2)

	feats += featsInc

	hconv3 = af(bn((conv2d(hconv2, weights[2]) + biases[2]),is_training=is_training))
	hconv3 = mp(hconv3,kS=mpkS,stride=2)

	feats += featsInc

	hconv4 = af(bn((conv2d(hconv3, weights[3]) + biases[3]),is_training=is_training))
	hconv4 = mp(hconv4,kS=mpkS,stride=2)

	feats += featsInc

	hconv5 = af(bn((conv2d(hconv4, weights[4]) + biases[4]),is_training=is_training))
	hconv5 = mp(hconv5,kS=mpkS,stride=2)

	feats += featsInc

	hconv6 = af(bn((conv2d(hconv5, weights[5]) + biases[5]),is_training=is_training))
	hconv6 = mp(hconv6,kS=mpkS,stride=2)

	feats += featsInc

	hconv7 = af(bn((conv2d(hconv6, weights[6]) + biases[6]),is_training=is_training))
	hconv7 = mp(hconv7,kS=mpkS,stride=2)

	#feats += featsInc
        
	#hconv8 = af(bn((conv2d(hconv7, weights[7]) + biases[7]),is_training=is_training))
	#hconv8 = mp(hconv8,kS=2,stride=2)
	#feats += featsInc

	sizeBeforeReshape = hconv7.get_shape().as_list()

	nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*sizeBeforeReshape[3]

	flatten = tf.reshape(hconv7, [-1, nFeats])
        flatten = tf.nn.dropout(flatten,keepProb)

	nLin1 = 128 
	wLin1 = weightVar([nFeats,nLin1])
	bLin1 = biasVar([nLin1])
	linear = af(bn(tf.matmul(flatten,wLin1) + bLin1,is_training=is_training))

	nLin2 = nClasses
	wLin2 = weightVar([nLin1,nLin2])
	bLin2 = biasVar([nLin2])
	yPred = tf.matmul(linear,wLin2) + bLin2

	for l in [x, hconv1,hconv2,hconv3,hconv4,hconv5,hconv6,hconv7,flatten,linear,yPred]:
	    print(l.get_shape())

	return yPred

def model2(x,inDims,nClasses,nFeatsInit,nFeatsInc,is_training):
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
            hconv = af(bn((conv2d(inTensor, w1) + b1),is_training=is_training))
            hconv = af(bn((conv2d(hconv, w2) + b2),is_training=is_training))
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


