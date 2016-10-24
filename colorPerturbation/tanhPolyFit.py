import numpy as np
import pandas as pd
from feeder import feeder
import ipdb
import os,glob
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline as spline
from scipy.optimize import curve_fit 

def mse(yPred,y):
	return np.square(y-yPred).mean()

def tanh(x,a,b,c,d,e,f,g,h):
	return a*np.tanh((x-b)/c) + d + e*np.arctanh((x-f)/g)  + h*x**3

def fitPoly(x,y,order,model="spline",saveFig=False,name="name"):
	if model == "poly":
		coeffs = np.polyfit(x,y,deg=order)
		model = np.poly1d(coeffs)
	elif model == "spline":
		model = spline(x,y,w=None,k=order,s=30,ext=3)
		coeffs = model.get_coeffs()
	elif model == "tanh":
		x[np.where(x<=0)] = 0.001
		x[np.where(x>=1)] = 1 - 0.001
		p0 = [0.4,0.4,0.13,0.52,0.03,0.00,1.2,0.2]
		coeffs, cov = curve_fit(tanh,x,y,p0=p0,maxfev=20000)

		model = lambda x: tanh(x,coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5],coeffs[6],coeffs[7]) #  coeffs[4],coeffs[5],coeffs[6],coeffs[7])
				

	yPred = model(x)
	loss = mse(yPred,y)
	if saveFig==True:
		assert name != None, "Please provide a name for plot"
		plt.title(loss)
		plt.scatter(x,y)
		plt.plot(x,yPred,"r")
		plt.savefig("plots/{0}.png".format(name))
		plt.close()
	return coeffs, loss

if __name__ == "__main__":

	import matplotlib.pyplot as plt

	nFits = 100
	modelType = "tanh"

	if modelType == "spline":
		print("Fitting {0} splines".format(nFits))
		order = 3 

	elif modelType == "poly":
		order = 5 
		print("Fitting {0} polynomials of size {1}".format(nFits,order))

	elif modelType == "tanh":
		order = 8 # number of params

	coeffs = np.zeros((nFits,order))
	plots = glob.glob("plots/*.png")
	[os.remove(plot) for plot in plots]

	losses = np.zeros(nFits)
	couldntFit = 0

	for i in tqdm(range(nFits)):
		feed =feeder(shuffle=False,pad=20)
		x,y = feed.next()
		if i % 1 == 0:
			saveFig = True
		else:
			saveFig = False
		try:
			coeffs[i], losses[i] = fitPoly(x=x,y=y,model=modelType,order=order,saveFig=saveFig,name=i)
		except RuntimeError:
			couldntFit += 1
			print("Run time error for {0}".format(couldntFit))
	pd.DataFrame(coeffs).to_csv("weights/poly{0}Weights.csv".format(order),index=0)
	ipdb.set_trace()

	losses = losses[~np.isnan(losses)]
	print("Average loss = {0}".format(losses.mean()))





