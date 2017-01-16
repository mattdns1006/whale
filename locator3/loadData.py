import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import json
import pandas as pd


def show(img,coords,sf):
    x1,y1,x2,y2 = [int(x*sf) for x in coords[0]]

    cv2.circle(img,(x1,y1),13,(255,0,0),10)
    cv2.circle(img,(x2,y2),13,(0,0,255),-1)
    plt.imshow(img)
    plt.show()

def readJson(fp):
    with open(fp) as f:
        data = json.load(f)
    return data

def makeCsv():
    df = pd.read_csv("../train.csv")
    imgsPath = "/home/msmith/kaggle/whale/imgs/"

    pfs = ["points1.json","points2.json"]
    dfOut = {}

    points1,points2 = readJson(pfs[0]), readJson(pfs[1])
    i = 0
    for p1 in tqdm(points1):
        assert len(p1["annotations"]) == 1
        fn = p1["filename"]
        for p2_ in points2:
            if p2_["filename"] == fn:
                p2 = p2_ # found corresponding point
                break

        path = df.whaleID[df.Image == p1["filename"]].values
        assert len(path) == 1, "more than one of same filename"
        path = path[0] + "/" + fn
        path = path.replace("w_","w1_")
        path = os.path.join(imgsPath,path)

        if not os.path.exists(path):
            print("{0} does not exist.".format(path))
            continue

        img = cv2.imread(path)
        h,w,c = img.shape

        x1 = p1["annotations"][0]["x"]/w
        y1 = p1["annotations"][0]["y"]/h
        x2 = p2["annotations"][0]["x"]/w
        y2 = p2["annotations"][0]["y"]/h
        dfOut[i] = [path,x1,y1,x2,y2,w,h]
        coords = dfOut[i][1:]
        #eg = cv2.resize(img,(800,800))
        #coords_ = [int(i*800) for i in [x1,y1,x2,y2]]
        i += 1
    dfOut = pd.DataFrame(dfOut).T
    dfOut.columns = ["path","x1","y1","x2","y2","w","h"]
    dfOut.to_csv("train.csv",index=0)
    print("Written path and corresponding points to train.csv")
    print(dfOut.head())


def showBatch(batchX,batchY,figsize=(15,15)):
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w)
    n, h, w, c = batchY.shape
    batchY = batchY.reshape(n*h,w)
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(batchX,cmap=cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(batchY,cmap=cm.gray)
    plt.show()

def prepImg(path,size):
    imageBytes = tf.read_file(path)
    decodedImg = tf.image.decode_jpeg(imageBytes)
    decodedImg = tf.image.resize_images(decodedImg,size)
    decodedImg = tf.cast(decodedImg,tf.float32)
    decodedImg = tf.mul(decodedImg,1/255.0)
    return decodedImg 

def read(csvPath,batchSize,inSize,shuffle):
    csv = tf.train.string_input_producer([csvPath],num_epochs=1,shuffle=shuffle)
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csv)
    defaults = [tf.constant([],dtype = tf.string),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.float32),
                tf.constant([],dtype = tf.int32),
                tf.constant([],dtype = tf.int32)]
    path, x1, y1, x2, y2, w, h = tf.decode_csv(v,record_defaults = defaults)
    coords = tf.pack([x1,y1,x2,y2])

    rs =  lambda x: tf.reshape(x,[1])
    x = prepImg(path,inSize)
    path = rs(path)
    inSizeC = list(inSize)
    inSizeC += [3]
    Q = tf.FIFOQueue(64,[tf.float32,tf.float32,tf.string],shapes=[inSizeC,[4],[1]])
    enQ = Q.enqueue([x,coords,path])
    QR = tf.train.QueueRunner(
            Q,
            [enQ]*8,
            Q.close(),
            Q.close(cancel_pending_enqueues=True)
            )
    tf.train.add_queue_runner(QR) 
    dQ = Q.dequeue()
    img,coords, imgPath = tf.train.batch(dQ,batchSize,16)
    return img, coords, imgPath 

if __name__ == "__main__":
    import pdb, cv2
    #makeCsv()

    sf = 800
    inSize = [sf,sf]
    img, coords = read(csvPath="train.csv",batchSize=1,inSize=inSize,shuffle=True)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.initialize_local_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        count = 0
        try:
            while True:
                out = sess.run([img,coords])
                x, y = out[0], out[1]

                pdb.set_trace()
                show(x[0],y,sf=sf)

                if coord.should_stop():
                    break
        except Exception,e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
