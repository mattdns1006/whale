import tensorflow as tf
import ipdb
import pandas as pd
import os
import matplotlib.pyplot as plt

def show(X,Y="none"):
    bs, h, w, c = X.shape 
    X = X.reshape(bs*h,w,c)[:,:,::-1]
    plt.imshow(X)
    plt.title(Y)
    plt.show()

def oneHot(idx,nClasses=447):
    oh = tf.sparse_to_dense(idx,output_shape = [nClasses], sparse_values = 1.0)
    return oh

if __name__ == "__main__":


    # Decode csv
    csvPath = "/home/msmith/kaggle/whale/trainCV.csv"
    df = pd.read_csv(csvPath)

    #csvPath = "/home/msmith/kaggle/whale/identifier/trainCV10.csv"
    print(df.head())
    print(df.shape)

    csvQ = tf.train.string_input_producer([csvPath])
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csvQ)

    defaults = [tf.constant([], shape = [1], dtype = tf.int32),
                tf.constant([], dtype = tf.string)]
                
    label, path = tf.decode_csv(v,record_defaults=defaults)

    labelOh = oneHot(idx=label)
    label = tf.reshape(label,[1])

    # Define subgraph to take filename, read filename, decode and enqueue
    image_bytes = tf.read_file(path)
    decoded_img = tf.image.decode_jpeg(image_bytes)
    #imageQ = tf.FIFOQueue(128,[tf.uint8,tf.float32,tf.string])
    imageQ = tf.FIFOQueue(128,[tf.uint8,tf.int32,tf.float32], shapes = [[600,800,3],[1],[447]])
    enQ_op = imageQ.enqueue([decoded_img,label,labelOh])

    NUM_THREADS = 16
    Q = tf.train.QueueRunner(
            imageQ,
            [enQ_op]*NUM_THREADS,
            imageQ.close(),
            imageQ.close(cancel_pending_enqueues=True)
            )

    tf.train.add_queue_runner(Q)
    bS = 4
    x,yIdx,y = imageQ.dequeue_many(bS)
    #x,y,path = imageQ.dequeue()


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        count = 0
        while not coord.should_stop():
            x_, y_, yIdx_ = sess.run([x,y,yIdx])
            ipdb.set_trace()
            show(x_,yIdx_)
            count += x_.shape[0]



