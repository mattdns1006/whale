import cv2, sys, glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.append("/home/msmith/misc/pcaRotate")
from pcaRotate import main as rotateCrop
from tqdm import tqdm

if __name__ == "__main__": 
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
	for path in tqdm(imgPaths):
		orig, mask = [cv2.imread(x)[:,:,::-1] for x in [path.replace("m1","w1"),path]]
		croppedHead, _, _= rotateCrop(orig,mask,ellipseThresh=20,redThresh=(0,60,250),outputWH=(400,300))
                wp = path.replace("m1","head")
                cv2.imwrite(wp,croppedHead)
                if numpy.random.uniform() < 0.01:
                    print(wp)
