from __future__ import division
import cv2
import numpy as np
from math import sqrt, pow, exp
from scipy.sparse import linalg
import h5py
import gc

if __name__ == "__main__":
	with h5py.File('weight_matrix.h5', 'r') as hf:
		W=np.array(hf.get('weight_matrix'))
	d=np.sum(W,1)
	D=np.diag(d)

	D = D.astype("float32",copy=False)
	W = W.astype("float32",copy=False)

	DW=np.subtract(D,W)

	del d
	del W

	print np.isfinite(DW).all()
	print np.isfinite(D).all()

	print (D.transpose() == D).all()
	print (DW.transpose() == DW).all()

	print np.linalg.det(DW)
	print np.linalg.det(D)

	eigVal,eigVec=linalg.eigs(A=DW,k=2,M=D,which='SM')
	print '\n'
	print eigVal
	print eigVec
	with h5py.File('eig.h5', 'w') as hf:
		hf.create_dataset('eigVal', data=eigVal)
		hf.create_dataset('eigVec', data=eigVec)
