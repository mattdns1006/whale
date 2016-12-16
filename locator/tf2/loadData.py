import tensorflow as tf
import pandas as pd
from numpy import random as rng
import os, glob
import matplotlib.pyplot as plt

def show(X,Y,figsize=(10,10)):
    bs, h, w, c = X.shape 
    X = X.reshape(bs*h,w,c)
    bs, h, w, c = Y.shape 
    Y = Y.reshape(bs*h,w,c)
    plt.figure(figsize=figsize) 
    plt.subplot(121)
    plt.imshow(X)
    plt.subplot(122)
    plt.imshow(Y)
    plt.show()

def prepImg(img,shape):
    img = tf.cast(img,tf.float32)
    img = tf.mul(img,1/255.0)
    img = tf.image.resize_images(img,shape[0],shape[1],method=0,align_corners=False)
    return img

def makeCsvs():
    train = glob.glob("../augmented/train/x_*")
    test = glob.glob("../augmented/testAugmented/x_*")
    testAug = glob.glob("../augmented/test/x_*")

    allPaths = train + test + testAug
    df = pd.DataFrame({"xPath":allPaths})
    yPath = lambda x: x.replace("x_","y_")
    df["yPath"] = df.xPath.apply(yPath)
    df.to_csv("train.csv",index=0)

    toFit = glob.glob("../../imgs/*/w1S_*")
    dfToFit = pd.DataFrame({"xPath":toFit})
    dfToFit.to_csv("toFit.csv",index=0)

def readCsv(csvPath,shuffle=False):
    if shuffle == True:
        print("Shuffling csv")
        shuffleCsvSave(csvPath)
    csvQ = tf.train.string_input_producer([csvPath])
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csvQ)

    defaults = [tf.constant([], dtype = tf.string),
                tf.constant([], dtype = tf.string)]
                
    xPath, yPath = tf.decode_csv(v,record_defaults=defaults)
    return xPath, yPath 

def decodeImagePath(path,shape):
    image_bytes = tf.read_file(path)
    decoded_img = tf.image.decode_jpeg(image_bytes)
    decoded_img = prepImg(decoded_img,shape=shape)
    return decoded_img
    
def loadData(csvPath,inShape,outShape,batchSize=10,batchCapacity=40,nThreads=16,shuffle=False): 
    xPath, yPath = readCsv(csvPath,shuffle=shuffle)
    xPathRe = tf.reshape(xPath,[1])

    # Define subgraph to take filenames, read filename, decode and enqueue
    xDecode, yDecode = decodeImagePath(xPath,inShape), decodeImagePath(yPath,outShape)
    imageQ = tf.FIFOQueue(128,[tf.float32,tf.float32,tf.string], shapes = [inShape,outShape,[1]])
    enQ_op = imageQ.enqueue([xDecode,yDecode,xPathRe])

    NUM_THREADS = nThreads
    Q = tf.train.QueueRunner(
            imageQ,
            [enQ_op]*NUM_THREADS,
            imageQ.close(),
            imageQ.close(cancel_pending_enqueues=True)
            )

    tf.train.add_queue_runner(Q)
    dQ = imageQ.dequeue()
    X, Y, paths = tf.train.batch(dQ, batch_size = batchSize, capacity = batchCapacity)
    return X, Y, paths


if __name__ == "__main__":
	import ipdb
	makeCsvs()
	inShape = [600,900,3]
	outShape = [38,57,3]

	x, y, paths = loadData("train.csv",inShape,outShape)
        init_op = tf.initialize_all_variables()
    	ipdb.set_trace()

	with tf.Session() as sess:

		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		while True:
			x_, y_, paths_ = sess.run([x,y,paths])
			show(x_,y_)
			ipdb.set_trace()

