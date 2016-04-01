import numpy as np
from scipy.sparse import linalg


def ncutValue(thresh,eigVecSmall,D,W,d):
	x=np.zeros_like(eigVecSmall)
	x.fill(-1)
	x[eigVecSmall>thresh]=1

	k=np.sum(d[x>0])/np.sum(d)
	b=k/(1-k)
	y=(1+x)-(b*(1-x))
	t1=np.dot(np.transpose(y),(D-W))
	num=np.dot(t1,y)
	t1=np.dot(np.transpose(y),D)
	den=np.dot(t1,y)
	ncut=num/den
	return ncut