import tensorflow as tf
import loadData
import model
import numpy as np
import glob, ipdb, sys 

sys.path.append("/home/msmith/misc/py/")
import matplotlib.pyplot as plt
from hStackBatch import hStackBatch

def showBatch(x,y,yPred,unnormalize=1):
	if unnormalize == 1:
		x *= 255
		y *= 255
		yPred *= 255

        x,y,yPred = [im.astype(np.uint8).squeeze() for im in [x,y,yPred]]
	
	plt.subplot(211)
	plt.imshow(x)
	plt.subplot(212)
	plt.imshow(np.vstack((y,yPred)))
	plt.show()


def mse(y,yPred):
	return tf.reduce_mean(tf.square(tf.sub(y,yPred)))

def dice(y,yPred):
    y_, yPred_ = tf.reshape(y), tf.reshape(yPred)
    return tf.matmul(y_,yPred_)

def trainer(loss,learningRate,momentum=0.9):
    return tf.train.MomentumOptimizer(learningRate,momentum).minimize(loss)

if __name__ == "__main__":

        import matplotlib.pyplot as plt

        lr = 0.1
        load = 0
	paths = glob.glob("../augmented/x_*")
	feeder = loadData.feed(inDims = [1,600,900,3], outDims = [1,19,29,3], paths = paths)

    	x, y, yPred = model.main()

        loss = mse(y,yPred)
        train = trainer(loss,lr)

        saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            if load == 1: 
                saver.restore(sess,"model.ckpt")

            for i in xrange(1000):
                X,Y = feeder.next()

                _, loss_, yPred_ = sess.run([train,loss,yPred], feed_dict = {x:X,y:Y})


                if i % 100 ==0:
                    showBatch(X,Y,yPred_)

                print(loss_)
            ipdb.set_trace()


        


