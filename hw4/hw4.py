import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering
import sys

data = pd.read_csv(sys.argv[2])
comp1 = data.iloc[:,1].astype('int')
comp2 = data.iloc[:,2].astype('int')

def test(label,m):
	

	with open(sys.argv[3],'w') as output:
		output.write('ID,Ans\n')
		for i in range(comp1.shape[0]):
			if label[comp1[i]] == label[comp2[i]]:
				output.write(str(i)+',1 \n')
				
			else:
				output.write(str(i)+',0 \n')
				
	





image = np.load(sys.argv[1]) 
image = image/255

m = 274
image_reduce = PCA(n_components=274, whiten=True , svd_solver='auto', random_state=0).fit_transform(image)

image_cluster = KMeans(n_clusters=2,random_state=1).fit(image_reduce)
label = image_cluster.labels_

test(label,m)

