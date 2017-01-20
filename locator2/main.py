import cv2,os,sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
from model import model0
sys.path.append("/Users/matt/misc/tfFunctions/")
from dice import dice

def showBatch(batchX,batchY,batchZ,wp,figsize=(15,15)):
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w,c)
    n, h, w, c = batchY.shape
    batchY = batchY.reshape(n*h,w,c)
    n, h, w, c = batchZ.shape
    batchZ = batchZ.reshape(n*h,w,c)
    X = np.hstack((batchX,batchY,batchZ))*255.0
    X = X[:,:,::-1]
    cv2.imwrite(wp,X)

def varSummary(var,name):
    with tf.name_scope('summary'):
        tf.summary.scalar(name, var)
        tf.summary.histogram(name, var)

def imgSummary(name,img):
    tf.summary.image(name,img)

def lossFn(y,yPred):
    with tf.variable_scope("loss"):
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y,yPred))))
    return loss

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,outSize,trainOrTest,initFeats,incFeats,nDown):
    if trainOrTest == "train":
        csvPath = "csvs/train.csv"
        print("Training")
        shuffle = True
    elif trainOrTest == "test":
        csvPath = "csvs/test.csv"
    X,Y,xPath = loadData.read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            outSize = outSize,
            shuffle=shuffle) #nodes

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
    flags.DEFINE_float("lr",0.0001,"Initial learning rate.")
    flags.DEFINE_float("lrD",1.00,"Learning rate division rate applied every epoch. (DEFAULT - nothing happens)")
    flags.DEFINE_integer("inSize",256,"Size of input image")
    flags.DEFINE_integer("outSize",256,"Size of output image")
    flags.DEFINE_integer("initFeats",16,"Initial number of features.")
    flags.DEFINE_integer("incFeats",0,"Number of features growing.")
    flags.DEFINE_integer("nDown",8,"Number of blocks going down.")
    flags.DEFINE_integer("bS",10,"Batch size.")
    flags.DEFINE_integer("load",0,"Load saved model.")
    flags.DEFINE_integer("trainAll",0,"Train on all data.")
    flags.DEFINE_integer("fit",0,"Fit training data.")
    flags.DEFINE_integer("fitTest",0,"Fit actual test data.")
    flags.DEFINE_integer("show",0,"Show for sanity.")
    flags.DEFINE_integer("nEpochs",10,"Number of epochs to train for.")
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
        for epoch in xrange(FLAGS.nEpochs):
            print("{0} of {1}".format(epoch,FLAGS.nEpochs))
            if epoch == FLAGS.nEpochs - 1 and FLAGS.trainAll == 0:
                print("Testing this time round")
                what = [tr,"test"]
            else:
                what = [tr]
            for trTe in what:
                if epoch > 0 or trTe == "test":
                    load = 1
                    tf.reset_default_graph()
            	saver,xPath,X,Y,YPred,loss,is_training,trainOp,learningRate = nodes(
                        batchSize=FLAGS.bS,
                        trainOrTest=trTe,
                        inSize = [FLAGS.inSize,FLAGS.inSize],
                        outSize = [FLAGS.outSize,FLAGS.outSize],
                        initFeats=FLAGS.initFeats,
                        incFeats=FLAGS.incFeats,
                        nDown=FLAGS.nDown
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
                                print("wtf r u doin")

                            if count % 200 == 0:
                                print("Epoch {0} seen {1} examples".format(epoch,count))
                                if FLAGS.show == 1:
                                    showBatch(x,y,yPred,"{0}epochNo_{1}.jpg".format(imgPath,epoch,count))



                            if coord.should_stop():
                                break
                    except Exception,e:
                        coord.request_stop(e)
                    finally:
                        coord.request_stop()
                        coord.join(threads)
                    print("Saving in {0}".format(savePath))
                    lrC = FLAGS.lr
                    FLAGS.lr /= FLAGS.lrD
                    print("Dropped learning rate from {0} to {1}".format(lrC,FLAGS.lr))
                    if trTe == "train":
                        print("Saving")
                        saver.save(sess,savePath)
                    sess.close()
                            
