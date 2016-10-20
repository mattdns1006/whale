import numpy as np
import tensorflow as tf
import histMatch
import ipdb
import glob,cv2 
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle

def plotBatch(batch):
	for i in range(batch.shape[0]):
		plt.plot(batch[i])
	plt.show(block=False)

def feeder(bS):
    fileNames = glob.glob("../imgs/*/head*")
    n = len(fileNames)
    X = np.arange(256).astype(np.float32)
    X /= 255.0
    c1, c2 = np.random.choice(n,2)
    while True:
        img1, img2 = cv2.imread(fileNames[c1]),cv2.imread(fileNames[c2])
        _, mapping = histMatch.histMatch(img1,img2)


        Y = mapping.astype(np.float32)# Thing we want to learn x = 0,1,...,255 y = cdf(x)

        Y /= 255.0
        XY = np.vstack((X,Y))

        rand = np.random.permutation(256)
        x, y = XY[:,rand][:,:bS].astype(np.float32)
        x, y = [np.expand_dims(arr,1) for arr in [x,y]]

	yield x,y

def placeHolder(bS):
	x = tf.placeholder(tf.float32,shape=(None,1))
	y = tf.placeholder(tf.float32,shape=(None,1))
	return x, y

def model(x):
	with tf.name_scope("tanhWeights"):
		t1 = tf.Variable(np.random.normal(1.0,scale=0.001))
		t2 = tf.Variable(np.random.normal(1.0,scale=0.001))
		t3 = tf.Variable(np.random.normal(1.0,scale=0.001))
		t4 = tf.Variable(np.random.normal(1.0,scale=0.001))
		yPred = t1*tf.tanh((x-t2)/t3) + t4
	return yPred

def loss(yPred,y):
	loss = tf.reduce_mean(tf.squared_difference(yPred,y))
	return loss

def train(loss,lr):
	optimizer = tf.train.AdamOptimizer(lr)
	trainOp = optimizer.minimize(loss)
	return trainOp

if __name__ == "__main__":

    bS = 50 

    x,y = placeHolder(bS)
    yPred = model(x)
    mse = loss(yPred,y)
    trainOp = train(mse,0.01)


    with tf.Session() as sess:


        losses = []
        saver = tf.train.Saver()
        for fit in tqdm(range(4)):
            init = tf.initialize_all_variables()
	    sess.run(init)
            feed = feeder(256)
            batchX, batchY = feed.next()
            for i in tqdm(range(100)):
                    _, mse_, yPred_ = sess.run([trainOp,mse,yPred],feed_dict = {x : batchX, y : batchY})
                    losses.append(mse_)

                    if i % 40 == 0:
                        print("Mean loss = {0}".format(np.array(losses).mean()))
                        losses = []
            

            saver.save(sess,"weights"+str(fit))
            fitX = np.linspace(0,1,256).reshape(256,1)
            fitY = yPred.eval(feed_dict={x:fitX})
            plt.plot(fitX,fitY,"o")
            plt.scatter(batchX,batchY)
            plt.title("Total loss = {0:.5f}".format(mse_))
            plt.show()








