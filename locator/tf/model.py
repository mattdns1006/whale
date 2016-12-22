import tensorflow as tf

import sys
sys.path.append("/home/msmith/misc/tfFunctions/")
import layers as layers
from tensorflow.contrib.layers import layers as tfLayers
from keras.layers.normalization import BatchNormalization

conv = layers.conv2d
deconv = layers.deconv2d
af = tf.nn.relu
af = tf.tanh
mp = layers.mp
bn = tfLayers.batch_norm

def placeholderX(batchSize=None,dims=(600,900,3),nDown=6):
    w,h,c = dims
    x = tf.placeholder(tf.float32,shape = [batchSize,w,h,c])

    return x

def modelUp(x,bS=10,nFeatsInc=32):
    W= {}
    B = {}
    feats = 32 
    nLayers = 7

    for i in range(nLayers):
        if i == 0:
            nIn = 3 
        elif i == nLayers-1:
            feats = 3
        W[i] = layers.weightVar([3,3,nIn,feats],stddev=0.35)
        B[i] = layers.biasVar([feats])

        nIn = feats
        feats += nFeatsInc 

    hConv1 = af(bn((conv(x,W[0]) + B[0]),is_training=True))
    hConv1 = mp(hConv1,3,2)

    hConv2 = af(bn(conv(hConv1,W[1]) + B[1],is_training=True))
    hConv2 = mp(hConv2,3,2)

    hConv3 = af(bn(conv(hConv2,W[2]) + B[2],is_training=True))
    hConv3 = mp(hConv3,3,2)

    hConv4 = af(bn(conv(hConv3,W[3]) + B[3],is_training=True))
    hConv4 = mp(hConv4,3,2)

    hConv5 = af(bn(conv(hConv4,W[4]) + B[4],is_training=True))
    hConv5 = mp(hConv5,2,2)
    hConv5 = af(bn(conv(hConv5,W[5]) + B[5],is_training=True))
    print(getShape(hConv5))

    hConv6 = af(bn(deconv(hConv5,layers.weightVar([5,5,96,192]),[bS,38,57,96],2)))
    hConv7 = af(bn(deconv(hConv6,layers.weightVar([5,5,96,96]),[bS,75,113,96],2)))
    hConv8 = hConv7 + hConv3
    hConv9 = bn(conv(hConv8,layers.weightVar([3,3,96,3],stddev=0.35)) + layers.biasVar([3]))
    yPred = tf.nn.sigmoid(hConv9)
    
    i = 1
    print("Model dims")
    for layer in [x,hConv1,hConv2,hConv3,hConv4,hConv5,hConv6,hConv7,hConv8,hConv9,yPred]:
	    print("Layer {0} = ".format(i),getShape(layer))
            i+=1

    return yPred

def model(x,batchSize,nFeatsInc=64):


    # Input = 900 x 600 x 3
    W= {}
    B = {}
    feats = 48 
    nLayers = 7

    for i in range(nLayers):
        if i == 0:
            nIn = 3 
        elif i == nLayers-1:
            feats = 3
        W[i] = layers.weightVar([3,3,nIn,feats],stddev=0.35)
        B[i] = layers.biasVar([feats])

        nIn = feats
        feats += nFeatsInc 

    hConv1 = af(bn((conv(x,W[0]) + B[0]),is_training=True))
    hConv1 = mp(hConv1,3,2)

    hConv2 = af(bn(conv(hConv1,W[1]) + B[1],is_training=True))
    hConv2 = mp(hConv2,3,2)

    hConv3 = af(bn(conv(hConv2,W[2]) + B[2],is_training=True))
    hConv3 = mp(hConv3,3,2)

    hConv4 = af(bn(conv(hConv3,W[3]) + B[3],is_training=True))
    hConv4 = mp(hConv4,3,2)

    hConv5 = bn(conv(hConv4,W[4]) + B[4],is_training=True)
    hConv5 = mp(hConv5,2,2)

    hConv6 = bn(conv(hConv5,W[5]) + B[5],is_training=True)

    yPred = tf.nn.sigmoid(hConv6)
    
    i = 1
    print("Model dims")
    for layer in [x,hConv1,hConv2,hConv3,hConv4,hConv5,hConv6,yPred]:
	    print("Layer {0} = ".format(i),getShape(layer))
            i+=1

    return yPred 

def getShape(tensor):
    return tensor.get_shape().as_list()

def aug(x):
    return tf.image.random_contrast(x,0.6,1.3)

def main(model,nFeatsInc = 64, batchSize=None,dims=(600,900,3),nDown=6):
    x = placeholderX(batchSize,dims,nDown)
    yPred = model(x,batchSize,nFeatsInc)
    oW, oH, oC = getShape(yPred)[1:]
    y = tf.placeholder(tf.float32,shape = [None,oW,oH,oC])
    return x, y,yPred


if __name__ == "__main__":
    import ipdb

    x = placeholderX(10,(600,900,3),6)
    x1 = modelUp(x)
    ipdb.set_trace()
    x_, y, yPred = main()


