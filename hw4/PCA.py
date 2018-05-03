import numpy as np
from skimage import io
import sys

file_path = sys.argv[1]
img_path = file_path + '/' + sys.argv[2]

def PCA():
	img = np.zeros((415,600,600,3))

	

	for i in range(415):
		img[i] = io.imread(file_path+'/'+str(i)+'.jpg')

	X = img.reshape(-1,600*600*3)
	X_mean = np.mean(X,axis=0)


	U, s, V = np.linalg.svd((X - X_mean).T, full_matrices=False)

	
	return U



def reconstruct(eigen_face):
	m = 4
	
	eigen_face = eigen_face.T
	
	choose = io.imread(img_path).flatten().astype(np.float)

	mean = np.mean(choose)
	weight = np.dot(eigen_face[:m,:],choose-mean)
	#print(np.load('s.npy'))
	#recover = eigen_face[2]
	recover = np.dot(weight,eigen_face[:m,:])+mean
	recover -= np.min(recover)
	recover /= np.max(recover)
	recover = (recover*255).astype(np.uint8)
	#io.imsave('original.jpg',io.imread(img_path))
	io.imsave('reconstruction.jpg', recover.reshape(600,600,3))






if __name__ == '__main__':
	U = PCA()
	reconstruct(U)



















