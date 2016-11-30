import matplotlib.pyplot as plt

def displayBatch(XY):
    X,Y = XY
    X *= 255.0
    names = train.decodeToName(Y)
    bs = X.shape[0]
    X = X.astype(np.uint8)
    fig = plt.figure()
    nRow = 2
    x = bs/nRow
    y = bs/x
    idx = 0
    ipdb.set_trace()
    for i in range(1,bs+1):
        ax = fig.add_subplot(bs+1,1,i)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.imshow(X[idx])
        ax.set_title(names[idx])
        idx +=1
    plt.show()
