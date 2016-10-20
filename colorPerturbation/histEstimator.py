import numpy as np
import pandas as pd
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
    img1, img2 = cv2.imread(fileNames[c1]),cv2.imread(fileNames[c2])
    _, mapping = histMatch.histMatch(img1,img2)

    Y = mapping.astype(np.float32)# Thing we want to learn x = 0,1,...,255 y = cdf(x)

    Y /= 255.0
    XY = np.vstack((X,Y))

    while True:
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
		t1 = tf.Variable(np.random.normal(0.4,scale=0.1))
		t2 = tf.Variable(np.random.normal(0.5,scale=0.1))
		t3 = tf.Variable(np.random.normal(0.2,scale=0.1))
		t4 = tf.Variable(np.random.normal(0.5,scale=0.01))
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

    np.random.seed(1006)
    bS = 10 
    nFits = int(100)
    nBatches = 30
    show = 1
    saveWeights = 0

    x,y = placeHolder(bS)
    yPred = model(x)
    lr = tf.constant(0.1,dtype=tf.float32)
    mse = loss(yPred,y)
    trainOp = train(mse,lr)
    import ipdb

    allWeights = np.zeros((nFits,4))

    with tf.Session() as sess:

        allLosses = np.zeros(nFits)
        saver = tf.train.Saver()
        for fit in tqdm(range(nFits)):
            losses = []
            init = tf.initialize_all_variables()

	    sess.run(init)
            feed = feeder(256)
            batchX, batchY = feed.next()
            learningRate = 0.07

            if show:
                #plt.subplot(121)
                fitX = np.linspace(0,1,256).reshape(256,1)
                fitY = yPred.eval(feed_dict={x:fitX})
                plt.plot(fitX,fitY,"r",label="init")
                init = [v.eval() for v in tf.trainable_variables()]


            for i in range(nBatches):
                    
                _, mse_, yPred_ = sess.run([trainOp,mse,yPred],feed_dict = {x : batchX, y : batchY, lr: learningRate})
                if i % 5 == 0 and i > 0: 
                    learningRate /= 1.1

                losses.append(mse_)
            


            allLosses[fit] = np.array(losses).mean()
            allWeights[fit] = [v.eval() for v in tf.trainable_variables()]

            if show:
                fitX = np.linspace(0,1,256).reshape(256,1)
                fitY = yPred.eval(feed_dict={x:fitX})
                plt.plot(fitX,fitY,"g",label="finalFit")
                plt.scatter(batchX,batchY,label="data")
                plt.legend(loc="upper left")
                plt.title("Total loss = {0:.5f}".format(mse_))
                #plt.subplot(122)
                #plt.plot(np.arange(nBatches),losses)
                plt.savefig("fits/{0}.png".format(fit))
                plt.close()
                

                #print("Initial weight values {0}".format(init))
                #print("Learning rate dropping to {0}".format(learningRate))
                #finalWeights = [v.eval() for v in tf.trainable_variables()]
                #print("Final weight values {0}".format(finalWeights))


            print("Average loss per fit = {0}".format(np.array(allLosses).mean()))
            #ipdb.set_trace()

    if saveWeights == 1:
        pd.DataFrame(allWeights).to_csv("weights/weights.csv",index=False)








