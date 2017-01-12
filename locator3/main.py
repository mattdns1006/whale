import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
import sys
from model import model0
sys.path.append("/Users/matt/misc/tfFunctions/")

def varSummary(var):
    with tf.name_scope('summary'):
        tf.summary.scalar('mean', var)
        tf.summary.histogram('mean', var)

def lossFn(y,yPred):
    return tf.reduce_mean(tf.square(tf.sub(y,yPred)))

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,trainOrTest):
    if trainOrTest == "train":
        csvPath = "train.csv"
        print("Training")
        shuffle = True
    elif trainOrTest == "test":
        print("Testing")
        csvPath = "testCV.csv"
        shuffle = False
    X,Y = loadData.read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            shuffle=shuffle) #nodes
    is_training = tf.placeholder(tf.bool)
    YPred = model0(X,is_training=is_training,nLayers=8,initFeats=16)
    with tf.variable_scope("loss"):
        loss = lossFn(Y,YPred)
    learningRate = tf.placeholder(tf.float32)
    trainOp = trainer(loss,learningRate)
    saver = tf.train.Saver()
    return saver,X,Y,YPred,loss,is_training,trainOp,learningRate

if __name__ == "__main__":
    import pdb
    inSize = [256,256]
    batchSize = 20
    nEpochs = 40
    load = 0

    savePath = "models/model0.tf"
    count = 0
    for epoch in xrange(nEpochs):
        print("{0} of {1}".format(epoch,nEpochs))
        if epoch > 0:
            load = 1
            tf.reset_default_graph()
        saver,X,Y,YPred,loss,is_training,trainOp,learningRate = nodes(batchSize=batchSize,inSize=inSize,trainOrTest="train")
        varSummary(loss)
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            if load == 1:
                saver.restore(sess,savePath)
            else:
                tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            train_writer = tf.summary.FileWriter("summary/train/",sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            try:
                while True:
                    _, summary = sess.run([trainOp,merged],feed_dict={is_training:True,learningRate:0.001})
                    count += batchSize
                    if count % 100 == 0:
                        print(count)
                    train_writer.add_summary(summary,count)
                    if coord.should_stop():
                        break
            except Exception,e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            saver.save(sess,savePath)
            sess.close()
