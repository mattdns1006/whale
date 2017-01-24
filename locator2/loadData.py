import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import json
import pandas as pd

def show(img):
    plt.imshow(img)
    plt.show()

def showBatch(batchX,batchY,figsize=(15,15)):
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w,c)
    n, h, w, c = batchY.shape
    batchY = batchY.reshape(n*h,w,c)
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(batchX,cmap=cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(batchY,cmap=cm.gray)
    plt.show()

def prepImg(path,size):
    imageBytes = tf.read_file(path)
    decodedImg = tf.image.decode_jpeg(imageBytes)
    decodedImg = tf.image.resize_images(decodedImg,size,0)
    decodedImg = tf.cast(decodedImg,tf.float32)
    decodedImg = tf.mul(decodedImg,1/255.0)
    decodedImg = tf.sub(decodedImg,tf.reduce_mean(decodedImg))
    return decodedImg 

def read(csvPath,batchSize,inSize,outSize,shuffle,num_epochs):
    csv = tf.train.string_input_producer([csvPath],num_epochs=num_epochs,shuffle=shuffle)
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csv)
    defaults = [tf.constant([],dtype = tf.string),
                tf.constant([],dtype = tf.string)]
    xPath, yPath = tf.decode_csv(v,record_defaults = defaults)

    rs =  lambda x: tf.reshape(x,[1])
    path = rs(xPath)
    x = prepImg(xPath,inSize)
    y = prepImg(yPath,outSize)

    inSizeC = list(inSize)
    outSizeC = list(outSize)
    inSizeC += [3]
    outSizeC += [3]

    Q = tf.FIFOQueue(64,[tf.float32,tf.float32,tf.string],shapes=[inSizeC,outSizeC,[1]])
    enQ = Q.enqueue([x,y,path])
    QR = tf.train.QueueRunner(
            Q,
            [enQ]*16,
            Q.close(),
            Q.close(cancel_pending_enqueues=True)
            )
    tf.train.add_queue_runner(QR) 
    dQ = Q.dequeue()
    X, Y, imgPath = tf.train.batch(dQ,batchSize,256,allow_smaller_final_batch=True)
    return X, Y, imgPath

if __name__ == "__main__":
    import pdb, cv2

    sf = 800
    inSize = [256,256]
    outSize = [256,256]
    out = read(csvPath="csvs/train.csv",batchSize=3,inSize=inSize,outSize=outSize,shuffle=True)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.initialize_local_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        count = 0
        try:
            while True:
                out_ = sess.run([out])
                out_ = out_[0]
                x, y = out_[0], out_[1]
                showBatch(x,y)
                pdb.set_trace()

                if coord.should_stop():
                    break
        except Exception,e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
