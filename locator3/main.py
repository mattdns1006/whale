import cv2,os,sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
from model import model0
sys.path.append("/Users/matt/misc/tfFunctions/")

def show(img,coords,sf,save,savePath):
    x1,y1,x2,y2 = [int(x*sf) for x in coords]
    cv2.circle(img,(x1,y1),3,(255,255,255),3)
    cv2.circle(img,(x2,y2),3,(0,0,255),-1)
    if save == 1:
        img*=255.0
        img=img[:,:,::-1].astype(np.uint8)
        cv2.imwrite(savePath,img)
    else:
        plt.imshow(img)
        plt.show()

def varSummary(var):
    with tf.name_scope('summary'):
        tf.summary.scalar('mean', var)
        tf.summary.histogram('mean', var)

def lossFn(y,yPred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y,yPred))))

def pixelLoss(y,yPred):
    return tf.reduce_mean(tf.abs(tf.sub(y,yPred)))*256 # regarding number of pixels

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,trainOrTest,initFeats,incFeats,sf,nDown):
    if trainOrTest == "train":
        csvPath = "trainCV.csv"
        print("Training")
        shuffle = True
    elif trainOrTest == "test":
        print("Testing")
        csvPath = "testCV.csv"
        shuffle = False
    elif trainOrTest == "fit":
        print("Fitting")
        csvPath = "train.csv"
        shuffle = True
    X,Y,xPath = loadData.read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            shuffle=shuffle) #nodes
    is_training = tf.placeholder(tf.bool)
    YPred = model0(X,is_training=is_training,nDown=nDown,initFeats=initFeats,featsInc=incFeats)
    with tf.variable_scope("loss"):
        loss = lossFn(Y,YPred)
        varSummary(loss)
    with tf.variable_scope("pixelLoss"):
        lossP= pixelLoss(Y,YPred)
        varSummary(lossP)
    learningRate = tf.placeholder(tf.float32)
    trainOp = trainer(loss,learningRate)
    saver = tf.train.Saver()
    return saver,xPath,X,Y,YPred,loss,lossP,is_training,trainOp,learningRate

if __name__ == "__main__":
    import pdb
    nEpochs = 10
    flags = tf.app.flags
    FLAGS = flags.FLAGS 
    flags.DEFINE_float("lr",0.0001,"Initial learning rate.")
    flags.DEFINE_integer("sf",256,"Size of input image")
    flags.DEFINE_integer("initFeats",64,"Initial number of features.")
    flags.DEFINE_integer("incFeats",16,"Number of features growing.")
    flags.DEFINE_integer("nDown",6,"Number of blocks going down.")
    flags.DEFINE_integer("bS",20,"Batch size.")
    flags.DEFINE_integer("load",0,"Load saved model.")
    flags.DEFINE_integer("fit",0,"Load saved model.")
    batchSize = FLAGS.bS
    load = FLAGS.load
    if FLAGS.fit == 1:
        load = 1
    specification = "{0}_{1}_{2}_{3}_{4}_{5}".format(FLAGS.sf,FLAGS.initFeats,FLAGS.incFeats,FLAGS.nDown,FLAGS.lr,FLAGS.bS)
    print("Specification = {0}".format(specification))
    savePath = "models/model0.tf"

    inSize = [FLAGS.sf,FLAGS.sf]
    trCount = teCount = 0
    if not FLAGS.fit:
        for epoch in xrange(nEpochs):
            print("{0} of {1}".format(epoch,nEpochs))
            for trTe in ["train","test"]:
                if epoch > 0 or trTe == "test":
                    load = 1
                    tf.reset_default_graph()
                saver,xPath,X,Y,YPred,loss,lossP,is_training,trainOp,learningRate = nodes(
                        batchSize=FLAGS.bS,
                        inSize=inSize,
                        trainOrTest=trTe,
                        initFeats=FLAGS.initFeats,
                        incFeats=FLAGS.incFeats,
                        nDown=FLAGS.nDown,
                        sf = FLAGS.sf
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
                    try:
                        while True:
                            if trTe == "train":
                                _, summary,x,y,yPred = sess.run([trainOp,merged,X,Y,YPred],feed_dict={is_training:True,learningRate:FLAGS.lr})
                                trCount += batchSize
                                trWriter.add_summary(summary,trCount)
                            
                            else:
                                summary,x,y,yPred = sess.run([merged,X,Y,YPred],feed_dict={is_training:False})
                                teCount += batchSize
                                teWriter.add_summary(summary,teCount)

                            if coord.should_stop():
                                break
                    except Exception,e:
                        coord.request_stop(e)
                    finally:
                        coord.request_stop()
                        coord.join(threads)
                    print("Saving in {0}".format(savePath))
                    saver.save(sess,savePath)
                    sess.close()
    else:
        fitBs
        fittedPath = "fitted/"+specification + "/"
        print("Saving files in {0}".format(fittedPath))
        if not os.path.exists(fittedPath):
            os.mkdir(fittedPath)
        saver,xPath,X,Y,YPred,loss,lossP,is_training,trainOp,learningRate = nodes(
                batchSize=fitBs,
                inSize=inSize,
                trainOrTest="fit",
                initFeats=FLAGS.initFeats,
                incFeats=FLAGS.incFeats,
                nDown=FLAGS.nDown,
                sf = FLAGS.sf
                )
        with tf.Session() as sess:
            count = 0
            print("Restoring {0}.".format(specification))
            saver.restore(sess,savePath)
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            try:
                while True:
                    xPath_, x,y,yPred = sess.run([xPath,X,Y,YPred],feed_dict={is_training:False})
                    count += fitBs 
                    bS = x.shape[0]
                    for i in xrange(bS):
                        fp = xPath_[i][0].split("/")[-1]
                        show(x[i],yPred[i],FLAGS.sf,1,fittedPath + fp)
                    if count % 200 == 0:
                        print(count)
                    if coord.should_stop():
                        break
            except Exception,e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            sess.close()
                            
