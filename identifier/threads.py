import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import tensorflow as tf
sys.path.insert(0,"/home/msmith/misc/py/")
import aug # Augmentation

def threadDequeue(nThreads,paths):
    filenameQ = tf.train.string_input_producer(paths)

    filename = filenameQ.dequeue()
    imageBytes = tf.read_file(filename)
    decodedImg = tf.image.decode_jpeg(imageBytes)
    imageQ = tf.FIFOQueue(128, [tf.uint8,tf.string], None)
    enqueueOp = imageQ.enqueue([decodedImg,filename])

    # Q runner
    queueRunner = tf.train.QueueRunner(
            imageQ,
            [enqueueOp] * nThreads,
            imageQ.close(),
            imageQ.close(cancel_pending_enqueues=True))

    tf.train.add_queue_runner(queueRunner)

    img,fname = imageQ.dequeue()
    return img, fname

def show(img,filename):
    plt.imshow(img)
    plt.title(filename)
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import ipdb, glob
    import numpy as np
    from tqdm import tqdm
    paths = glob.glob("../imgs/whale_33195*/head_ss_*")

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:

        sess.run(init_op)
        img,fname = threadDequeue(12,paths)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        count = 0
        try: 
            while not coord.should_stop(): 

                img_,fn_ = sess.run([img,fname])
                print(count)
                count +=1
                #ipdb.set_trace()
                #show(img_,fn_)
        except tf.errors.OutOfRangeError:
            print("Done training")
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

