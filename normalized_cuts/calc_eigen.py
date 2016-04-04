from __future__ import division
import cv2
import numpy as np
from math import sqrt, pow, exp
from scipy.sparse import linalg
import h5py

def eigen(W):
	d=np.sum(W,1)
	D=np.diag(d)

	D = D.astype("float32",copy=False)
	W = W.astype("float32",copy=False)

	DW=np.subtract(D,W)

	del d
	del W

	eigVal,eigVec=linalg.eigs(A=DW,k=2,M=D,which='SM')
	return eigVec
