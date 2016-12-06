import cv2, sys, glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.append("/home/msmith/misc/pcaRotate")
from pcaRotate import main as rotateCrop
from tqdm import tqdm
import argparse

if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Process arg.')
	parser.add_argument('ss', metavar='SS', type=int, help='Semi supervised mask?')
	args = parser.parse_args()
	import ipdb,numpy
	def show(img,gray=0):
		if gray ==1:
		    plt.imshow(img,cmap=cm.gray)
		else:
		    plt.imshow(img)
		plt.show()
	imgPaths = glob.glob("/home/msmith/kaggle/whale/imgs/whale*/m1*")
	testPaths = glob.glob("/home/msmith/kaggle/whale/imgs/test*/m1*")
	print("Make sure image is in RGB order not BGR")
	imgPaths.sort()
	maskPath = "m1"
	dstPath = "head"
        if args.ss == 1:
            maskPath += "_ss"
            dstPath += "_ss"
	for path in tqdm(imgPaths):
		path = path.replace("m1",maskPath)
		orig, mask = [cv2.imread(x)[:,:,::-1] for x in [path.replace(maskPath,"w1"),path]]
		croppedHead, _, _= rotateCrop(orig,mask,ellipseThresh=20,redThresh=(0,60,250),outputWH=(400,300))
                wp = path.replace(maskPath,dstPath)
                cv2.imwrite(wp,croppedHead)
                if numpy.random.uniform() < 0.01:
                    print(wp)
