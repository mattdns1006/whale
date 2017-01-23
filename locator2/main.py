import cv2,os,sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
from denseNet import model0
sys.path.append("/Users/matt/misc/tfFunctions/")
from dice import dice

def showBatch(batchX,batchY,batchZ,wp,figsize=(15,15)):
    outSize = 100 # width
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w,c)
    n, h, w, c = batchY.shape
    batchY = batchY.reshape(n*h,w,c)
    n, h, w, c = batchZ.shape
    batchZ = batchZ.reshape(n*h,w,c)
    Y = np.hstack((batchY,batchZ))
    Y = Y[:,:,::-1]
    Y = cv2.resize(Y,(outSize*2,outSize*n))
    X = cv2.resize(batchX,(outSize,outSize*n))[:,:,::-1]
    out = np.hstack((X,Y))*255.0
    cv2.imwrite(wp.replace(".jpg","eg.jpg"),out)


def varSummary(var,name):
    with tf.name_scope('summary'):
        tf.summary.scalar(name, var)
        tf.summary.histogram(name, var)

def imgSummary(name,img):
    tf.summary.image(name,img)

def lossFn(y,yPred):
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(tf.sub(y,yPred)))
    return loss

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,outSize,trainOrTest,initFeats,incFeats,nDown,num_epochs):
    if trainOrTest == "train":
        csvPath = "csvs/trainS.csv"
        print("Training")
        shuffle = True
    elif trainOrTest == "test":
        csvPath = "csvs/test.csv"
        num_epochs = 1
        shuffle = False 
    X,Y,xPath = loadData.read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            outSize = outSize,
            shuffle=shuffle,
            num_epochs = num_epochs
            ) #nodes

    is_training = tf.placeholder(tf.bool)
    YPred = model0(X,is_training=is_training,nDown=nDown,initFeats=initFeats,featsInc=incFeats)
    loss = lossFn(Y,YPred)
    #diceScore = dice(YPred,Y)
    #varSummary(diceScore,"dice")
    learningRate = tf.placeholder(tf.float32)
    trainOp = trainer(loss,learningRate)
    saver = tf.train.Saver()
    varSummary(loss,"loss")

    #imgSummary("X",X)
    #imgSummary("Y",Y)
    #imgSummary("yPred",YPred)
    return saver,xPath,X,Y,YPred,loss,is_training,trainOp,learningRate

if __name__ == "__main__":
    import pdb
    nEpochs = 20
    flags = tf.app.flags
    FLAGS = flags.FLAGS 
    flags.DEFINE_float("lr",0.01,"Initial learning rate.")
    flags.DEFINE_float("lrD",1.00,"Learning rate division rate applied every epoch. (DEFAULT - nothing happens)")
    flags.DEFINE_integer("inSize",512,"Size of input image")
    flags.DEFINE_integer("outSize",64,"Size of output image")
    flags.DEFINE_integer("initFeats",16,"Initial number of features.")
    flags.DEFINE_integer("incFeats",16,"Number of features growing.")
    flags.DEFINE_integer("nDown",3,"Number of blocks going down.")
    flags.DEFINE_integer("bS",10,"Batch size.")
    flags.DEFINE_integer("load",0,"Load saved model.")
    flags.DEFINE_integer("trainAll",0,"Train on all data.")
    flags.DEFINE_integer("fit",0,"Fit training data.")
    flags.DEFINE_integer("fitTest",0,"Fit actual test data.")
    flags.DEFINE_integer("show",0,"Show for sanity.")
    flags.DEFINE_integer("nEpochs",200,"Number of epochs to train for.")
    batchSize = FLAGS.bS
    load = FLAGS.load
    if FLAGS.fit == 1 or FLAGS.fitTest == 1:
        load = 1
    specification = "{0}_{1:.6f}_{2}_{3}_{4}_{5}_{6}".format(FLAGS.bS,FLAGS.lr,FLAGS.inSize,FLAGS.outSize,FLAGS.initFeats,FLAGS.incFeats,FLAGS.nDown)
    print("Specification = {0}".format(specification))
    modelDir = "models/" + specification + "/"
    imgPath = modelDir + "imgs/"
    if not FLAGS.fit:
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)
            os.mkdir(imgPath)
    savePath = modelDir + "model.tf"
    trCount = teCount = 0
    tr = "train"
    if FLAGS.fit == 0 and FLAGS.fitTest == 0:
        for trTe in ["train"]:
            if trTe == "test":
                load = 1
                tf.reset_default_graph()
            saver,xPath,X,Y,YPred,loss,is_training,trainOp,learningRate = nodes(
                    batchSize=FLAGS.bS,
                    trainOrTest=trTe,
                    inSize = [FLAGS.inSize,FLAGS.inSize],
                    outSize = [FLAGS.outSize,FLAGS.outSize],
                    initFeats=FLAGS.initFeats,
                    incFeats=FLAGS.incFeats,
                    nDown=FLAGS.nDown,
                    num_epochs=FLAGS.nEpochs
                    )
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

            merged = tf.summary.merge_all()
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                if load == 1:
                    print("Restoring {0}.".format(specification))
                    saver.restore(sess,savePath)
                else:
                    tf.global_variables_initializer().run()
                tf.local_variables_initializer().run()
                trWriter = tf.summary.FileWriter("summary/{0}/train/".format(specification),sess.graph)
                teWriter = tf.summary.FileWriter("summary/{0}/test/".format(specification),sess.graph)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                count = 0
                try:
                    while True:
                        if trTe in ["train","trainAll"]:
                            _, summary,x,y,yPred = sess.run([trainOp,merged,X,Y,YPred],feed_dict={is_training:True,learningRate:FLAGS.lr})
                            trCount += batchSize
                            count += batchSize
                            trWriter.add_summary(summary,trCount)
                        
                        elif trTe == ["test"]:
                            summary,x,y,yPred = sess.run([merged,X,Y,YPred],feed_dict={is_training:False})
                            teCount += batchSize
                            teWriter.add_summary(summary,teCount)
                        else:
                            break

                        if count % 50 == 0:
                            print("Seen {0} examples".format(count))
                            if FLAGS.show == 1:
                                showBatch(x,y,yPred,"{0}eg.jpg".format(imgPath))

                        if coord.should_stop():
                            break
                except Exception,e:
                    coord.request_stop(e)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                print("Finished! Seen {0} examples".format(count))
                print("Saving in {0}".format(savePath))
                lrC = FLAGS.lr
                FLAGS.lr /= FLAGS.lrD
                print("Dropped learning rate from {0} to {1}".format(lrC,FLAGS.lr))
                if trTe == "train":
                    print("Saving")
                    saver.save(sess,savePath)
                sess.close()
                        
