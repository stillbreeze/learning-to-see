import h5py
import numpy as np
import calc_ncut as ncut
import calc_eigen as eig
import scipy.optimize
import cv2



def NcutRecursive(W,segFinal,imgSeg,threshcut,thresharea):

	eigVec=eig.eigen(W)
	d=np.sum(W,1)
	D=np.diag(d)

	D=D.astype("float32",copy=False)
	W=W.astype("float32",copy=False)

	eigVecSmall=eigVec[:,1]
	eigVecSmall=eigVecSmall.real
	thresh=np.mean(eigVecSmall)

	thresh=scipy.optimize.fmin(ncut.ncutValue, thresh, args=(eigVecSmall, D, W, d))[0]

	A=np.where(eigVecSmall>thresh)
	B=np.where(eigVecSmall<=thresh)

	print A
	print '\n'
	print B

	if (ncut.ncutValue(thresh,eigVecSmall,D,W,d) > threshcut) or (A.shape[0]<thresharea) or (B.shape[0]<thresharea):
		return 1
	segFinal.append((A,B))
	res=NcutRecursive(np.take(W[A],A,1),segFinal,np.take(imgSeg,A),threshcut,thresharea)
	res=NcutRecursive(np.take(W[B],A,1),segFinal,np.take(imgSeg,B),threshcut,thresharea)
	print 'Writing segments'
	with h5py.File('segFinal.h5', 'w') as hf:
		hf.create_dataset('segFinal', data=segFinal)

if __name__ == "__main__":
	img=cv2.imread('./colors.jpg',0)
	imgSeg=cv2.resize(img,(50,50))

	with h5py.File('weight_matrix.h5', 'r') as hf:
		W=np.array(hf.get('weight_matrix'))
	segFinal=[]
	threshcut=0.2
	thresharea=13
	NcutRecursive(W,segFinal,imgSeg,threshcut,thresharea)