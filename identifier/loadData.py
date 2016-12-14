import tensorflow as tf
import pandas as pd
from numpy import random as rng
import os
import matplotlib.pyplot as plt

def show(X,Y="none"):
    bs, h, w, c = X.shape 
    X = X.reshape(bs*h,w,c)[:,:,::-1]
    plt.imshow(X)
    plt.title(Y)
    plt.show()

def prepImg(img,shape):
    img = tf.cast(img,tf.float32)
    img = tf.mul(img,1/255.0)
    img = tf.image.resize_images(img,shape[0],shape[1],method=0,align_corners=False)
    return img

def oneHot(idx,nClasses=447):
    oh = tf.sparse_to_dense(idx,output_shape = [nClasses], sparse_values = 1.0)
    return oh

def shuffleCsvSave(csvPath):
    csv = pd.read_csv(csvPath)
    nObs = csv.shape[0]
    rIdx = rng.permutation(nObs)
    csv = csv.reindex(rIdx)
    csv.reset_index(drop=1,inplace=1)
    csv.to_csv(csvPath,index=0)

def readCsv(csvPath,shuffle=False):
    if shuffle == True:
        print("Shuffling csv")
        shuffleCsvSave(csvPath)
    csvQ = tf.train.string_input_producer([csvPath])
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csvQ)

    defaults = [tf.constant([], shape = [1], dtype = tf.int32),
                tf.constant([], dtype = tf.string)]
                
    label, path = tf.decode_csv(v,record_defaults=defaults)
    return path, label

def loadData(csvPath,shape, batchSize=10,batchCapacity=40,nThreads=16,shuffle=False): 
    path, label = readCsv(csvPath,shuffle=shuffle)
    labelOh = oneHot(idx=label)
    pathRe = tf.reshape(path,[1])

    # Define subgraph to take filename, read filename, decode and enqueue
    image_bytes = tf.read_file(path)
    decoded_img = tf.image.decode_jpeg(image_bytes)
    decoded_img = prepImg(decoded_img,shape=shape)
    imageQ = tf.FIFOQueue(128,[tf.float32,tf.float32,tf.string], shapes = [shape,[447],[1]])
    enQ_op = imageQ.enqueue([decoded_img,labelOh,pathRe])

    NUM_THREADS = nThreads
    Q = tf.train.QueueRunner(
            imageQ,
            [enQ_op]*NUM_THREADS,
            imageQ.close(),
            imageQ.close(cancel_pending_enqueues=True)
            )

    tf.train.add_queue_runner(Q)
    dQ = imageQ.dequeue()
    X, Y, Ypaths = tf.train.batch(dQ, batch_size = batchSize, capacity = batchCapacity)
    return X, Y, Ypaths

if __name__ == "__main__":

    # Decode csv
    csvPathTr = "/home/msmith/kaggle/whale/trainCV.csv"
    csvPathTe = "/home/msmith/kaggle/whale/testCV.csv"
    shape = [560/2,400/2,3]
    bs = 4
    Xtr, Ytr, YtrPaths = loadData(csvPathTr,shape=shape,batchSize=bs)
    Xte, Yte, YtePaths = loadData(csvPathTe,shape=shape,batchSize=bs)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        count = 0
        for i in xrange(10000):
            x_, y_, Ypaths_ = sess.run([Xtr, Ytr, YtrPaths])
            #show(x_,Ypaths_)
            count += x_.shape[0]
            print(count)

        count = 0
        for i in xrange(10):
            x_, y_, Ypaths_ = sess.run([Xte, Yte, YtePaths])
            show(x_,Ypaths_)
            count += x_.shape[0]
            print(count)



