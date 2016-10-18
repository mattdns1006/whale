import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob, cv2
from pylab import rcParams
rcParams["figure.figsize"] = 25,10

def show(img,gray=0):
    if gray == 1:
        plt.imshow(img,cmap = cm.gray); plt.show();
    else:
        plt.imshow(img); plt.show();

def brgToYuv(img):
    ''' Convert to YUV '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def yuvToBrg(img):
    ''' Convert back to BRG '''
    return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

def cdfImg(img):
    ''' Returns Cumulative distribution x and F(X)'''
    imFlat = img.flatten()
    x, counts = np.unique(imFlat, return_counts=True)
    cdfx = np.cumsum(counts)
    return x, cdfx

def findNearest(array,value):
    ''' Finds nearest value in array and returns the value and its index in the array'''
    idx = (np.abs(array.astype(np.float32)-value)).argmin()
    return array[idx], idx

def histMatch(img1,img2,plot=0):
    ''' Image 1 is converted to have same CDF as image 2 '''
    x1, cdf1 = cdfImg(img1)
    x2, cdf2 = cdfImg(img2)
    mapping = np.empty([2**8])

    if plot == 1:
        plt.plot(x1,cdf1,label = "image")
        plt.plot(x2,cdf2,label = "target")

    uint8 = np.arange(mapping.shape[0])
    for g1 in uint8:
        _,x1Nearest = findNearest(x1,g1)
        f1g1 = cdf1[x1Nearest] # Find CDF1 of current value
        f2g2, f2g2I = findNearest(cdf2,f1g1) # return index of nearest value in CDF2
        g2 = x2[f2g2I]

        mapping[g1] = g2

        if g1 % 50 == 0 and plot == 1:
            print("g1,F1g1 = {0},{1}, g2,F2g2={2},{3}".format(g1,f1g1,g2,f2g2))
            plt.plot([g1,g1],[0,f1g1],color = "yellow", linestyle='--',linewidth=3)
            plt.plot([g1,g2],[f1g1,f2g2],linestyle='--',color = "red",linewidth=3)
            plt.plot([g2,g2],[0,f2g2],linestyle='--',color = "orange",linewidth=3)


    im1YT = mapping[img1].astype(np.uint8) # recompute
    x1YT, cdf1YT = cdfImg(im1YT)

    if plot == 1:
        plt.plot(x1YT,cdf1YT,label = ["transformed"])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show(block=False)

    return im1YT, mapping

def getDst(img1,img2): 

    ''' Wrapper function given two images we match the hist of the first image to the second for each color channel in YUV'''

    yuv1, yuv2 = [brgToYuv(x) for x in [img1,img2]]
    dst = np.zeros(yuv1.shape) # Init our final image
    mappings = np.zeros((3,256))

    for chan in range(3): # for YUV

        
        c1, c2 = [x[:,:,chan] for x in [yuv1, yuv2]]
        dst[:,:,chan], mappings[chan] = histMatch(c1,c2)

    dst = yuvToBrg(dst.astype(np.uint8)) ## Convert back to normal
    return dst, mappings


if __name__ == "__main__":
    import ipdb


    np.random.seed(1006)
    ## Get random whales
    head = glob.glob("../imgs/*/head_*")
    while True:
        i,j = np.random.randint(0,len(head),2)
        img1 = cv2.imread(head[i])
        img2 = cv2.imread(head[j])
        dst,mappings = getDst(img1,img2) 

        plt.subplot(311)
        plt.title("Im1/Im2 with average pixel values = {0:0.2f}/{1:0.2f}".format(np.mean(img1),np.mean(img2)))
        plt.imshow(np.hstack((img1,dst,img2)))
        plt.subplot(312)
        x1, im1Cdf = cdfImg(img1[:,:,0])
        x2, im2Cdf = cdfImg(img2[:,:,0])
        xDst, dstCdf = cdfImg(dst[:,:,0])
        plt.plot(x1,im1Cdf,label = ["source"])
        plt.plot(x2,im2Cdf,label = ["target"])
        plt.plot(xDst,dstCdf,label = ["dest"])
        plt.subplot(313)
        for i in range(3):
            plt.plot(np.arange(mappings[i].shape[0]),mappings[i],label = ["mapping" + str(i)])

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()





