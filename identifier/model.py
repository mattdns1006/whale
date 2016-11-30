import tensorflow as tf
import sys
sys.path.insert(0,"/home/msmith/misc/tfFunctions/")

from layers import *
from tensorflow.contrib.layers import layers as tfLayers
bn = tfLayers.batch_norm
af = tf.nn.relu

def model1(x,inDims,nClasses):
	bs, h, w, c = inDims
	weights = {}
	biases = {}
	feats = 32 
	featsInc = 32 
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

	hconv1 = af(bn((conv2d(x, weights[0]) + biases[0]),is_training=True))
	hconv1 = mp(hconv1,kS=2,stride=2)

	feats += featsInc

	hconv2 = af(bn((conv2d(hconv1, weights[1]) + biases[1]),is_training=True))
	hconv2 = mp(hconv2,kS=2,stride=2)

	feats += featsInc

	hconv3 = af(bn((conv2d(hconv2, weights[2]) + biases[2]),is_training=True))
	hconv3 = mp(hconv3,kS=2,stride=2)

	feats += featsInc

	hconv4 = af(bn((conv2d(hconv3, weights[3]) + biases[3]),is_training=True))
	hconv4 = mp(hconv4,kS=2,stride=2)

	feats += featsInc

	hconv5 = af(bn((conv2d(hconv4, weights[4]) + biases[4]),is_training=True))
	hconv5 = mp(hconv5,kS=2,stride=2)

	feats += featsInc

	hconv6 = af(bn((conv2d(hconv5, weights[5]) + biases[5]),is_training=True))
	hconv6 = mp(hconv6,kS=2,stride=2)

	feats += featsInc

	hconv7 = af(bn((conv2d(hconv6, weights[6]) + biases[6]),is_training=True))
	hconv7 = mp(hconv7,kS=2,stride=2)

	feats += featsInc

	sizeBeforeReshape = hconv7.get_shape().as_list()

	nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*sizeBeforeReshape[3]

	flatten = tf.reshape(hconv7, [-1, nFeats])

	nLin = 512
	wLin1 = weightVar([nFeats,nLin])
	bLin1 = biasVar([nLin])
	fc1 = af(bn(((tf.matmul(flatten,wLin1) + bLin1)),is_training=True))

	wLin2 = weightVar([nLin,nClasses])
	bLin2 = biasVar([nClasses])

	yPred = tf.nn.softmax(tf.matmul(fc1,wLin2) + bLin2)
	yPred = tf.clip_by_value(yPred,1e-10,1.0)

	for l in [x, hconv1,hconv2,hconv3,hconv4,hconv5,hconv6,hconv7,flatten,fc1,yPred]:
	    print(l.get_shape())

	return yPred

