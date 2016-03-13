from __future__ import division
import cv2
import numpy as np
from math import sqrt, pow, exp
from scipy.sparse import csr_matrix
import h5py


def calcNorm(i,j,img,W_i,sigmaI,r):
	row=img.shape[0]
	col=img.shape[1]
	for x in xrange(row):
		for y in xrange(col):
			dist=sqrt(pow((i-x),2)+pow((j-y),2))
			if dist<r:
				temp=exp((-pow(cv2.norm(img[i][j],img[x][y],cv2.NORM_L1),2))/sigmaI)
				W_i[i+j][x+y]=temp

def spatialAffinity(img,r):
	pass

def intensityAffinity(img,r):
	row=img.shape[0]
	col=img.shape[1]
	N=row*col
	print N
	W_i=np.zeros((N,N))
	print W_i.shape
	(means, stds) = cv2.meanStdDev(img)	
	sigmaI=2*(pow(stds[0][0],2)+pow(stds[1][0],2)+pow(stds[2][0],2))
	for i in xrange(row):
		for j in xrange(col):
			calcNorm(i,j,img,W_i,sigmaI,r)
			print i,j
	with h5py.File('W_i.h5', 'w') as hf:
	    hf.create_dataset('W_i', data=W_i)

if __name__ == "__main__":
	img=cv2.imread('./colors.jpg',3)
	img=cv2.resize(img,(150,150))
	cv2.imshow('',img)
	r=2.5
	intensityAffinity(img,r)
	spatialAffinity(img,r)
	cv2.waitKey(0)
	cv2.destroyAllWindows()