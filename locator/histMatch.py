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
    idx = (np.abs(array-value)).argmin()
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

    return im1YT

def main(img1,img2): 
    ''' Wrapper function given two images we match the hist of the first image to the second for each color channel in YUV'''


if __name__ == "__main__":
    import ipdb
    head = glob.glob("../imgs/*/head_*")
    i,j = np.random.random_integers(0,len(head),2)
    img1 = cv2.imread(head[i])
    img2 = cv2.imread(head[j])
    im1, im2 = [brgToYuv(x) for x in [img1,img2]]
    im1Y, im2Y = [x[:,:,0] for x in [im1, im2]]

    im1YT = histMatch(im1Y,im2Y)
    x1YT, cdf1YT = cdfImg(im1YT)
    im1C = im1.copy()
    im1C[:,:,0] = im1YT
    img1T = yuvToBrg(im1C)

    plt.subplot(311)
    plt.title("Im1/Im2 with average pixel values = {0:0.2f}/{1:0.2f}".format(np.mean(im1),np.mean(im2)))
    plt.imshow(np.hstack((img1,img1T,img2)))
    plt.subplot(312)
    plt.title("Y - luminance")
    plt.imshow(np.hstack((im1Y,im1YT,im2Y)),cmap=cm.gray)
    plt.subplot(313)
    x1, im1Cdf = cdfImg(im1Y)
    x2, im2Cdf = cdfImg(im2Y)
    plt.plot(x1,im1Cdf,label = ["im1"])
    plt.plot(x2,im2Cdf,label = ["im2"])
    plt.plot(x1YT,cdf1YT,label = ["im1X"])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()




