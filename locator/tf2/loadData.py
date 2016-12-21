import tensorflow as tf
import pandas as pd
from numpy import random as rng
import os, glob
import matplotlib.pyplot as plt

def prepImg(img,shape):
    img = tf.cast(img,tf.float32)
    img = tf.mul(img,1/255.0)
    img = tf.image.resize_images(img,shape[0],shape[1],method=0,align_corners=False)
    return img

def makeCsvs():
    train = glob.glob("../augmented/train/x_*")
    testAug = glob.glob("../augmented/testAugmented/x_*")
    test = glob.glob("../augmented/test/x_*")

    def trte(trOrTe):
        assert trOrTe in ["train","test","all"], "needs to be one of train or test or all"
        if trOrTe == "train":
            trOrTe = train
            wp = "train"
        elif trOrTe == "test":
            trOrTe = test 
            wp = "test"
        elif trOrTe == "all":
            trOrTe = train + testAug + test 
            wp = "all"
        df = pd.DataFrame({"xPath":trOrTe})
        yPath = lambda x: x.replace("x_","y_")
        df["yPath"] = df.xPath.apply(yPath)
        df.to_csv("{0}.csv".format(wp),index=0)
    [trte(x) for x in ["train", "test","all"]]

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
	inShape = [866,1300,3]
	outShape = [28,41,3]
	x, y, paths = loadData("test.csv",inShape,outShape)
        init_op = tf.initialize_all_variables()
    	ipdb.set_trace()

	with tf.Session() as sess:
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                count = 0
                ps = []

                while True:
                    try:
			x_, y_, paths_ = sess.run([x,y,paths])

                        count += 10
                        if count % 200 == 0:
                            ipdb.set_trace()
                        ps.append(paths_)
                    except tf.errors.OutOfRangeError:
                        print("finished")

