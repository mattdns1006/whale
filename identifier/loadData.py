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
    pathRe = tf.reshape(path,[1])

    # Define subgraph to take filename, read filename, decode and enqueue
    image_bytes = tf.read_file(path)
    decoded_img = tf.image.decode_jpeg(image_bytes)
    imageQ = tf.FIFOQueue(128,[tf.uint8,tf.int32,tf.float32,tf.string], shapes = [[600,800,3],[1],[447],[1]])
    enQ_op = imageQ.enqueue([decoded_img,label,labelOh,pathRe])

    NUM_THREADS = 16
    Q = tf.train.QueueRunner(
            imageQ,
            [enQ_op]*NUM_THREADS,
            imageQ.close(),
            imageQ.close(cancel_pending_enqueues=True)
            )

    tf.train.add_queue_runner(Q)
    bS = 4
    dQ = imageQ.dequeue()
    X, Ylab, Y, paths = tf.train.batch(dQ, batch_size = 10, capacity = 40)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        count = 0
        for i in xrange(10):
            x_, y_, yIdx_, paths_ = sess.run([X,Y,Ylab,paths])
            show(x_,paths_)
            count += x_.shape[0]
            print(count)




