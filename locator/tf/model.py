import tensorflow as tf
import sys
sys.path.append("/home/msmith/misc/tfFunctions/")
import layers as layers
from tensorflow.contrib.layers import layers as tfLayers
from keras.layers.normalization import BatchNormalization

def placeholderX(batchSize=None,dims=(600,900,3),nDown=6):
    w,h,c = dims
    x = tf.placeholder(tf.float32,shape = [batchSize,w,h,c])
    return x


def model(x):
    conv = layers.conv2d
    af = tf.nn.relu
    mp = layers.mp
    bn = tfLayers.batch_norm

    # Input = 900 x 600 x 3
    W= {}
    B = {}
    feats = 32
    nLayers = 6

    for i in range(nLayers):
        if i == 0:
            nIn = 3 
        elif i == nLayers-1:
            feats = 3
        W[i] = layers.weightVar([3,3,nIn,feats],stddev=0.35)
        B[i] = layers.biasVar([feats])

        nIn = feats
        feats += 0 

    hConv1 = af(bn((conv(x,W[0]) + B[0]),is_training=True))
    hConv1 = mp(hConv1,2,2)

    hConv2 = af(bn(conv(hConv1,W[1]) + B[1],is_training=True))
    hConv2 = mp(hConv2,2,2)

    hConv3 = af(bn(conv(hConv2,W[2]) + B[2],is_training=True))
    hConv3 = mp(hConv3,2,2)

    hConv4 = af(bn(conv(hConv3,W[3]) + B[3],is_training=True))
    hConv4 = mp(hConv4,2,2)

    hConv5 = af(bn(conv(hConv4,W[4]) + B[4],is_training=True))
    hConv5 = mp(hConv5,2,2)

    hConv6 = bn(conv(hConv5,W[5]) + B[5],is_training=True)
    yPred = tf.nn.sigmoid(hConv6)
    
    for layer in [hConv1,hConv2,hConv3,hConv4,hConv5,hConv6,yPred]:
	    print(getShape(layer))




    return yPred 

def getShape(tensor):
    return tensor.get_shape().as_list()

def main():
    x = placeholderX()
    yPred = model(x)
    oW, oH, oC = getShape(yPred)[1:]
    y = tf.placeholder(tf.float32,shape = [None,oW,oH,oC])
    return x,y,yPred


if __name__ == "__main__":
    x, y, yPred = main()


