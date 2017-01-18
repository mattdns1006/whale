import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import json
import pandas as pd


def show(img,coords,sf):
    x1,y1,x2,y2 = [int(x*sf) for x in coords[0]]

    cv2.circle(img,(x1,y1),13,(255,0,0),10)
    cv2.circle(img,(x2,y2),13,(0,0,255),-1)
    plt.imshow(img)
    plt.show()

def showBatch(batchX,batchY,figsize=(15,15)):
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w)
    n, h, w, c = batchY.shape
    batchY = batchY.reshape(n*h,w)
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(batchX,cmap=cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(batchY,cmap=cm.gray)
    plt.show()

def prepImg(path,size):
    imageBytes = tf.read_file(path)
    decodedImg = tf.image.decode_jpeg(imageBytes)
    decodedImg = tf.image.resize_images(decodedImg,size)
    decodedImg = tf.cast(decodedImg,tf.float32)
    decodedImg = tf.mul(decodedImg,1/255.0)
    return decodedImg 

def read(csvPath,batchSize,inSize,shuffle):
    csv = tf.train.string_input_producer([csvPath],num_epochs=1,shuffle=shuffle)
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csv)
    defaults = [tf.constant([],dtype = tf.string),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.int32),
                tf.constant([],dtype = tf.int32)]
    path, x1, y1, x2, y2, w, h = tf.decode_csv(v,record_defaults = defaults)
    coords = tf.pack([x1,y1,x2,y2])

    rs =  lambda x: tf.reshape(x,[1])
    x = prepImg(path,inSize)
    path = rs(path)
    inSizeC = list(inSize)
    inSizeC += [3]
    Q = tf.FIFOQueue(32,[tf.float32,tf.float32,tf.string],shapes=[inSizeC,[4],[1]])
    enQ = Q.enqueue([x,coords,path])
    QR = tf.train.QueueRunner(
            Q,
            [enQ]*8,
            Q.close(),
            Q.close(cancel_pending_enqueues=True)
            )
    tf.train.add_queue_runner(QR) 
    dQ = Q.dequeue()
    img,coords, imgPath = tf.train.batch(dQ,batchSize,16)
    return img, coords, imgPath 

if __name__ == "__main__":
    import pdb, cv2
    makeCsv()

    def foo():
        sf = 800
        inSize = [sf,sf]
        img, coords = read(csvPath="train.csv",batchSize=1,inSize=inSize,shuffle=True)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            tf.initialize_local_variables().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            count = 0
            try:
                while True:
                    out = sess.run([img,coords])
                    x, y = out[0], out[1]

                    pdb.set_trace()
                    show(x[0],y,sf=sf)

                    if coord.should_stop():
                        break
            except Exception,e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
