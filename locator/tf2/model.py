import tensorflow as tf
import sys
sys.path.append("/home/msmith/misc/tfFunctions/")
import layers as layers
from tensorflow.contrib.layers import layers as tfLayers
from keras.layers.normalization import BatchNormalization

def model1(x,nFeats,nFeatsInc=0):
    conv = layers.conv2d
    af = tf.nn.relu
    af = tf.tanh
    mp = layers.mp
    bn = tfLayers.batch_norm
    up = layers.deconv2d

    # Input = 900 x 600 x 3
    W= {}
    B = {}
    feats = 48 
    nLayers = 5

    for i in range(nLayers):
        if i == 0:
            nIn = 3 
        elif i == nLayers-1:
            feats = 3
        W[i] = layers.weightVar([3,3,nIn,feats],stddev=0.05)
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
    #hConv5 = mp(hConv5,2,2)

    #hConv6 = bn(conv(hConv5,W[5]) + B[5],is_training=True)

   # 
   # W = layers.weightVar([3,3,3,3])
   # B = layers.biasVar([3])
   # os = getShape(hConv6)
   # os[1]*=2
   # os[2]*=2
   # up1 = up(hConv6,W,os,2)

    yPred = tf.nn.sigmoid(hConv5)
    
    i = 1
    print("Model dims")
    for layer in [x,hConv1,hConv2,hConv3,hConv4,hConv5,yPred]:
	    print("Layer {0} = ".format(i),getShape(layer))
            i+=1

    return yPred 

def getShape(tensor):
    return tensor.get_shape().as_list()


if __name__ == "__main__":
    import ipdb
    x = tf.placeholder(tf.float32,[1,1300,866,3])
    yPred = model1(x)
    ipdb.set_trace()
    print("ere")


