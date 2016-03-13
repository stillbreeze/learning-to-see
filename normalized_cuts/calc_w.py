from __future__ import division
import cv2
import numpy as np
from math import sqrt, pow, exp
from scipy.sparse import csr_matrix
import h5py


def calcNorm(i,j,img,weight_matrix,sigmaI,sigmaS,r):
	row=img.shape[0]
	col=img.shape[1]
	for x in xrange(row):
		for y in xrange(col):
			dist=sqrt(pow((i-x),2)+pow((j-y),2))
			if dist<r:
				temp_s=exp((-pow(cv2.norm((i,j),(x,y),cv2.NORM_L2),2)))/pow(sigmaS,2)
				temp_i=exp((-pow(cv2.norm(img[i][j],img[x][y],cv2.NORM_L2),2)))/pow(sigmaI,2)
				weight_matrix[i+j][x+y]=temp_s*temp_i

def affinityMatrix(img,r):
	row=img.shape[0]
	col=img.shape[1]
	N=row*col
	weight_matrix=np.zeros((N,N))
	weight_matrix=weight_matrix.astype("float16",copy=False)
	# (means, stds)=cv2.meanStdDev(img)	
	# sigmaI=2*(pow(stds[0][0],2)+pow(stds[1][0],2)+pow(stds[2][0],2))
	sigmaI=500
	sigmaS=4
	for i in xrange(row):
		for j in xrange(col):
			calcNorm(i,j,img,weight_matrix,sigmaI,sigmaS,r)
			print i,j
	with h5py.File('weight_matrix.h5', 'w') as hf:
		hf.create_dataset('weight_matrix', data=weight_matrix)

if __name__ == "__main__":
	img=cv2.imread('./colors.jpg',3)
	img=cv2.resize(img,(150,150))
	r=2.5
	affinityMatrix(img,r)
	cv2.waitKey(0)
	cv2.destroyAllWindows()