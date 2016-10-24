import numpy as np
import pandas as pd
import tensorflow as tf
import histMatch
import glob,cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
from tensorflow.contrib.layers import batch_norm as bn
from feeder import feeder

def nWeights(trainableVars):
    from operator import mul
    count = 0
    for v in trainableVars:
        w = v.get_shape().as_list()
        if len(w) == 0:
            count += 1
        else:
            count += reduce(mul,w)
    return count

def atanh(x):
    return 0.5*tf.log((1+x)/(1-x))

def plotBatch(batch):
	for i in range(batch.shape[0]):
		plt.plot(batch[i])
	plt.show(block=False)



def placeHolder(bS):
	x = tf.placeholder(tf.float32,shape=(None,1))
	y = tf.placeholder(tf.float32,shape=(None,1))
        return x,y

def model1(x):
	with tf.name_scope("tanhWeights"):
                var = 0.2
                # Tanh 
		t1 = tf.Variable(np.random.normal(0.4,scale=var))
		t2 = tf.Variable(np.random.normal(0.5,scale=var))
		t3 = tf.Variable(np.random.normal(0.2,scale=var))
		t4 = tf.Variable(np.random.normal(0.5,scale=var))
		h1 = t1*tf.tanh((x-t2)/t3) + t4

                # Inverse tanh
		a1 = tf.Variable(np.random.normal(0.4,scale=var))
		a2 = tf.Variable(np.random.normal(0.2,scale=var))
		a3 = tf.Variable(np.random.normal(1.0,scale=var))
		a4 = tf.Variable(np.random.normal(0.2,scale=var))
                h2 = a1*atanh((x-a2)/a3) + a4

                # Linear combination of these tanhs yPred = w1h1 + w2h2
		w1 = tf.Variable(np.random.normal(0.3,scale=var))
		w2 = tf.Variable(np.random.normal(0.3,scale=var))
		w3 = tf.Variable(np.random.normal(0.3,scale=var))
		b1 = tf.Variable(np.random.normal(1.2,scale=var))

                yPred = w1*h1 + w2*h2 + w3*h1*h2 + b1 

	return yPred

def model2(x):
	with tf.name_scope("poly"):
            var = 0.01
            #x_ = tf.sigmoid(x)
            x_ = x
            '''
            y = a + bx + cx^2 + dx^3 + ... + zx^n
            for cdf...
            y(0) = 0
            y(1) = 1
            hence 
            a = 0
            1 = (b + c + d + ... + z)
            b = (1 - c - d - ... - z)
            there fore optimize (a to f);
            y = (1-c-d-e-f)x + cx^2 + dx^3 + ex^4 + fx^5
            '''
            order = 30 
            W = tf.Variable(tf.random_normal([order-1],mean=0,stddev=0.01))
            #b = (1 - tf.reduce_sum(W))
            b = tf.Variable(tf.random_normal([1],mean=0,stddev=0.01))
            yPred = tf.Variable(0.0)
            yPred = tf.add(yPred,b*x_)
            for n in xrange(2, order+1):
                yPred = tf.add(yPred,tf.mul(tf.pow(x_,n),W[n-2]))


        return yPred

def model3(x):
        nParams = 15 
        stddev = 0.5

        W = tf.Variable(tf.random_normal([1,nParams],mean=0,stddev=stddev))
        b = tf.Variable(tf.zeros(nParams))
        h1 = tf.nn.tanh(bn(tf.matmul(x,W) + b))

        W2 = tf.Variable(tf.random_normal([nParams,nParams],mean=0,stddev=stddev))
        b2 = tf.Variable(tf.zeros(nParams))
        h2 = tf.nn.tanh(bn(tf.matmul(h1,W2) + b2))
        
        W3 = tf.Variable(tf.random_normal([nParams,nParams],mean=0,stddev=stddev))
        b3 = tf.Variable(tf.zeros(nParams))
        h3 = tf.nn.tanh(bn(tf.matmul(h2,W3) + b3))

        W4 = tf.Variable(tf.random_normal([nParams,1],mean=0,stddev=stddev))
        b4 = tf.Variable(tf.zeros(1))
        yPred = tf.matmul(h3,W4) + b4
        return yPred

def mae(yPred,y):
	loss = tf.contrib.losses.absolute_difference(yPred,y)
	return loss

def mse(yPred,y):
	loss = tf.reduce_mean(tf.squared_difference(yPred,y))
	return loss

def ce(yPred,y):
	loss = -tf.reduce_mean(tf.log(yPred)*y)
	return loss

def train(loss,lr):
	optimizer = tf.train.MomentumOptimizer(lr,0.96)
	trainOp = optimizer.minimize(loss)
	return trainOp

if __name__ == "__main__":

    import ipdb

    np.random.seed(1006)
    bS = 256 
    nFits = int(40)
    nBatches = 300
    show = 1
    saveWeights = 0

    x,y = placeHolder(bS)
    yPred = model2(x)
    lr = tf.constant(0.001,dtype=tf.float32)
    lossFn = mse(yPred,y)
    trainOp = train(lossFn,lr)
    
    nTrVars = len(tf.trainable_variables())
    print("Number of weights = {0}".format(nWeights(tf.trainable_variables())))
    allWeights = np.zeros((nFits,nTrVars))

    with tf.Session() as sess:

        allLosses = np.zeros(nFits)
        saver = tf.train.Saver()
        for fit in tqdm(range(nFits)):
            
            init = tf.initialize_all_variables()

	    sess.run(init)
            feed = feeder(bS)
            learningRate = 0.1

            losses = np.zeros(nBatches)
            for i in range(nBatches):
                batchX, batchY = feed.next()
                _, loss, yPred_ = sess.run([trainOp,lossFn,yPred],feed_dict = {x : batchX, y : batchY, lr: learningRate})
                if i % 5 == 0 and i > 0: 
                    learningRate /= 1.05
                losses[i] = loss

            allLosses[fit] = losses[-1]
            #allWeights[fit] = [v.eval() for v in tf.trainable_variables()]

            if show:
                plt.subplot(121)
                fitX = np.linspace(0,1,256).reshape(256,1)
                fitY = yPred.eval(feed_dict={x:fitX})
                plt.plot(fitX,fitY,"g",label="finalFit")
                plt.scatter(batchX,batchY,label="data")
                plt.legend(loc="upper left")
                plt.title("Total loss = {0:.5f}".format(loss))
                plt.subplot(122)
                plt.plot(np.arange(nBatches),losses)
                plt.savefig("fits/{0}.png".format(fit))
                #plt.show()
                plt.close()
                
                #print("Learning rate dropping to {0}".format(learningRate))
                finalWeights = [v.eval() for v in tf.trainable_variables()]
                print("Final weight values {0}".format(finalWeights))


            avLoss = allLosses[np.where(allLosses!=0)].mean()
            print("Average loss per fit = {0}".format(avLoss))

    if saveWeights == 1:
        pd.DataFrame(allWeights).to_csv("weights/weights.csv",index=False)








