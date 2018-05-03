

import pandas as pd
import numpy as np
import sys
from keras.models import Sequential,load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Conv2D, Dropout, AveragePooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import regularizers
from scipy.stats import mode
def read_data():
	raw = pd.read_csv("train.csv")

	label = raw.iloc[:,0]
	data = raw.iloc[:,1]
	
	
	data = data.str.split(expand = True).astype('float32').values
	

	return data.reshape(-1,48,48,1),np_utils.to_categorical(label, 7)

def val(data,label):
	num = label.shape[0]
	index = np.random.permutation(num)
	x = data[:int(num*0.9)]
	y = label[:int(num*0.9)]
	val_x = data[int(num*0.9):]
	val_y = label[int(num*0.9):]
	return x,y,val_x,val_y








def test():
	raw = pd.read_csv("test.csv")
	data = raw.iloc[:,1]
	data = data.str.split(expand = True).astype('float32').values
	data = data.reshape(-1,48,48,1)
	
	model_list = []

	for i in range(20):
		model_list.append(load_model('model'+str(i)+'.h5'))
	ans = []
	pre = np.zeros((20,data.shape[0]),dtype=np.int32)
	
	for i in range(20):
		pre[i] = np.argmax(model_list[i].predict(data),axis=1)

	for i in range(data.shape[0]):
		ans.append(np.argmax(np.bincount(pre[:,i])))



	with open('result.csv','w') as output:
		output.write('id,label\n')
		for i in range(data.shape[0]):
			output.write(str(i)+','+str(ans[i])+'\n')


if __name__ == '__main__':
	data, label = read_data()
	train_x,train_y,val_x,val_y = val(data,label)
	#print(train_x.shape)
	#training(train_x,train_y,val_x,val_y)
	test()
