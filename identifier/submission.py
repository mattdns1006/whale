import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os, sys, glob
from tqdm import tqdm
from loadData import loadData
from model import model1, model2
import matplotlib.pyplot as plt

def yPredArgMax(yPred,n=3):
    '''
    Given single batch array returns position of top n predictions
    '''
    yPred = yPred.squeeze() # works on batch size of 1
    indicies = yPred.argsort()[-n:][::-1]
    values = yPred[indicies]
    return indicies, values

def show(x,yPred,name):
    plt.imshow(x[:,:,::-1])
    topPreds = yPredArgMax(yPred,n=5)
    print(topPreds)
    plt.title(name)
    plt.show()

def loadEncoding():
    import pickle
    encoding = pickle.load(open("../encoding.p","rb"))
    return encoding

def makeSubmissionCsv(ss=0):
    '''
    Make submission csv file
    '''
    if ss==1:
        ext = "ss"
    else:
        ext = "[0-9]"
    testFiles = glob.glob("../imgs/test/head_{0}*".format(ext))
    testFiles.sort()
    df = pd.DataFrame({"label":np.zeros(len(testFiles)).astype(np.int32),"fullPath":testFiles})
    df = df[["label","fullPath"]]
    df.to_csv("submission.csv",index=0)
    print("Made submission csv with {0} observations to test.".format(df.shape[0]))

def main():
    nFeatsInit = 32
    nFeatsInc = 32
    modelName = "models/model1.tf"
    bs = 10
    if os.path.exists("submission.csv") == False: 
        print("No submission csv file to process...")
        print("Making one...")
        makeSubmissionCsv()
    x, y, yPaths = loadData("submission.csv",shape=[300,300,3],batchSize=bs,batchCapacity=40,nThreads=16,shuffle=False)
    yPred = model1(x,inDims=[1,300,300,3],nClasses=447,nFeatsInit=nFeatsInit,nFeatsInc=nFeatsInc,keepProb=1.0)
    yPred = tf.nn.softmax(yPred)
    saver = tf.train.Saver()
    whaleDict = loadEncoding()
    header = sorted(whaleDict.keys())

    preds = np.empty((0,447))
    names = np.empty((0,1))
    with tf.Session() as sess:
        saver.restore(sess,modelName)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        count = 0
        while True:
    
            x_, y_, yPred_, yPaths_ = sess.run([x, y, yPred, yPaths])
            preds = np.vstack((preds,yPred_))
            names = np.vstack((names,yPaths_))
            #for i in xrange(x_.shape[0]):
            #    show(x_[i],yPred_[i],yPaths_[i])
            count += bs
            if count % 200 == 0:
                print(count)
            if count > 7000:
                break
        ipdb.set_trace()
        print("Finished")



if __name__ == "__main__":
    import ipdb
    main()
